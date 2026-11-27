/**
 * @file CUDAMoEKernels.cu
 * @brief CUDA MoE grouped NativeQuantised decode, floating MoE, canonical rank bank, and retained HC/FFN kernels.
 */
#include "CUDAMoEDecodeRouterBody.cuh"
#include "CUDAMoENativeQuantisedDecodeHelpers.cuh"

namespace
{
    __global__ void grouped_swiglu_quantize_blockwise_kernel(
        const float *const *__restrict__ gate_ptrs,
        const float *const *__restrict__ up_ptrs,
        const int *__restrict__ expert_ids,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int num_active,
        int K)
    {
        const int slot = blockIdx.x;
        if (slot >= num_active)
            return;
        if (expert_ids && expert_ids[slot] < 0)
            return;

        constexpr int kBlockSize = 32;
        constexpr int kWarps = 8;
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int blocks_per_row = K / kBlockSize;
        const float *gate = gate_ptrs[slot];
        const float *up = up_ptrs[slot];
        int8_t *row_int8 = A_int8 + static_cast<size_t>(slot) * K;
        float *row_scales = scales_A_blockwise + static_cast<size_t>(slot) * blocks_per_row;

        for (int block_idx = warp_id; block_idx < blocks_per_row; block_idx += kWarps)
        {
            const int block_start = block_idx * kBlockSize;
            publish_decode_swiglu_q8(gate[block_start + lane], up[block_start + lane],
                row_int8 + block_start, row_scales + block_idx, lane);
        }
    }


    /**
     * @brief Split-K scatter kernel for grouped gate/up decode projection.
     *
     * Each block reduces one K-partition of one output column for one expert
     * slot, writing its partial to a [slot][k_part][N] scratch buffer. A separate
     * reduce kernel sums the partials. This raises occupancy versus the single
     * full-K kernel (which launches only num_active * ceil(N/64) blocks) by
     * multiplying the block count by k_partitions, exposing far more warps to
     * hide global-memory latency on the weight payload loads.
     *
     * Grid:  ((N + 63)/64, k_partitions, num_active)   Block: (64)
     * @tparam CodebookId Physical prepared decode layout.
     * @tparam FixedQFN Use the admitted fixed-pitch and fixed-partition schedule.
     */
    template <uint8_t CodebookId, bool FixedQFN = false>
    __global__ void grouped_native_quantised_gate_up_kpart_decode_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int slot = blockIdx.z;
        constexpr bool PackedQFN = FixedQFN && !kQFNDecodeGateUpCodebook<CodebookId>;
        if constexpr (PackedQFN)
        {
            // The measured packed entry submits only complete column and K
            // tiles. Route and descriptor resolution precede the dot below.
            static_assert(kQFNDecodeGateUpN % kTileN == 0);
            N = kQFNDecodeGateUpN;
            K = kQFNDecodeGateUpK;
            k_partitions = llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        }
        else if (slot >= num_active || k_part >= k_partitions || n >= N)
            return;

        // Linear index into the [num_active][k_partitions][N] partials buffer.
        const size_t partial_index =
            (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);

        const int expert_id = expert_ids[slot];
        if (expert_id < 0)
            return;
        assert(expert_id < num_experts);
        if (expert_id >= num_experts)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        // Evenly split the K-blocks across the k_partitions; this block owns
        // [b_start, b_end).
        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;

        if (b_start >= b_end)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
        if constexpr (FixedQFN)
        {
            static_assert(kQFNDecodeFixedGateUpCodebook<CodebookId>);
            using Traits = llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;
            constexpr auto order = [] {
                if constexpr (!PackedQFN)
                    return NativeMoEDotOrder::PartitionPartial;
                if constexpr (Traits::is_iq_grid)
                    return NativeMoEDotOrder::VerifierWideGrid;
                return NativeMoEDotOrder::VerifierK32;
            }();
            constexpr bool split_iq1_s = CodebookId == native_formats::IQ1_S.codebook_id;
            constexpr auto signs = CodebookId == native_formats::IQ2_XS.codebook_id
                ? NativeMoEIQ2SignSchedule::ClampedFunnelAndIndependentProducts
                : NativeMoEIQ2SignSchedule::RegisterShifts;
            float gate = 0.0f;
            float up = 0.0f;
            if (native_quantised_pair_shape_ok<CodebookId>(gate_desc, up_desc, N, K))
                native_quantised_gate_up_fixed_partition<CodebookId, kQFNDecodeGateUpN, kQFNDecodeGateUpPackets,
                    order, split_iq1_s, signs>(
                    gate_desc, up_desc, A_int8, scales_A_blockwise, n, b_start, gate, up);
            gate_partials[partial_index] = gate;
            up_partials[partial_index] = up;
            return;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
        // Q4_K, Q5_K and Q8_0 share one activation traversal while retaining
        // each role's independent ordered FP32 accumulation.
        if constexpr (!FixedQFN && kQFNDecodeGateUpCodebook<CodebookId>)
        {
            float gate = 0.0f;
            float up = 0.0f;
            native_quantised_dot_desc_pair_range<CodebookId>(
                gate_desc, up_desc, n, A_int8, scales_A_blockwise, N, K,
                b_start, b_end, gate, up);
            gate_partials[partial_index] = gate;
            up_partials[partial_index] = up;
            return;
        }
#endif
        gate_partials[partial_index] = native_quantised_dot_desc_range_dispatch<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        up_partials[partial_index] = native_quantised_dot_desc_range_dispatch<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
    }

    /**
     * @brief Reduce K-partition partials into FP32 gate/up or the exact Down packet.
     *
     * Sums the k_partitions partial contributions for each (slot, n) produced by
     * grouped_native_quantised_gate_up_kpart_decode_kernel and writes the result to
     * the per-slot gate/up output buffers. With direct publication each warp
     * instead passes the same two FP32 sums to the original SwiGLU quantizer.
     *
     * Grid: ((N + 63)/64, num_active)   Block: (64)
     * @tparam PublishQ8 Whether the prepared consumer requires complete K32 packets.
     * @param gate_partials Original slot-major ordered gate partials.
     * @param up_partials Original slot-major ordered up partials.
     * @param expert_ids Original slot IDs, with negative IDs for inactive slots.
     * @param gate_outputs Separate staged FP32 gate rows when requested.
     * @param up_outputs Separate staged FP32 up rows when requested.
     * @param num_active Number of original route slots.
     * @param N Original local intermediate width.
     * @param k_partitions Fixed numerical reduction count.
     * @param swiglu Direct publication's original slot-major Q8 planes.
     */
    template<bool PublishQ8>
    __global__ void grouped_native_quantised_gate_up_kpart_reduce_kernel(
        const float *__restrict__ gate_partials,
        const float *__restrict__ up_partials,
        const int *__restrict__ expert_ids,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int N,
        int k_partitions, llaminar2::CUDAMoESwiGLUPublication swiglu)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int slot = blockIdx.y;
        if (slot >= num_active || n >= N)
            return;
        if (expert_ids && expert_ids[slot] < 0)
            return;

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        const size_t slot_base =
            static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
        for (int k_part = 0; k_part < k_partitions; ++k_part)
        {
            const size_t idx =
                slot_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
            gate_sum += gate_partials[idx];
            up_sum += up_partials[idx];
        }

        if constexpr (PublishQ8)
        {
            // Launch admission requires N % 32 == 0, so no live warp diverges.
            const int lane = threadIdx.x & 31;
            const auto index = static_cast<size_t>(slot) * N + n - lane;
            publish_decode_swiglu_q8(gate_sum, up_sum,
                swiglu.codes + index, swiglu.scales + index / 32, lane);
        }
        else
        {
            gate_outputs[slot][n] = gate_sum;
            up_outputs[slot][n] = up_sum;
        }
    }

    /**
     * @brief Compute one unweighted split-K partial per original router slot.
     *
     * Unlike the ordinary decode kernel, this launch never sums different
     * routes.  The route dimension survives the participant collective, which
     * makes expert ownership irrelevant to the eventual FP32 addition tree.
     */
    template <uint8_t CodebookId>
    __global__ void grouped_native_quantised_down_kpart_decode_route_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_ids,
        float *__restrict__ partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int route = blockIdx.z;
        if (route >= num_active || k_part >= k_partitions || n >= N)
            return;

        const size_t partial_index =
            (static_cast<size_t>(route) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);
        const int expert_id = expert_ids[route];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        const int blocks_per_row = K / 32;
        const int blocks_per_part =
            (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        const int b_end = min(blocks_per_row, b_start + blocks_per_part);
        if (b_start >= b_end)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        const DeviceNativeQuantisedMatrixDesc desc = descs[expert_id];
        const int8_t *slot_A = A_int8 + static_cast<size_t>(route) * K;
        const float *slot_scales =
            scales_A_blockwise + static_cast<size_t>(route) * blocks_per_row;
        partials[partial_index] =
            native_quantised_dot_desc_range_dispatch<CodebookId>(
                desc, n, slot_A, slot_scales, N, K, b_start, b_end);
    }

    /**
     * @brief Publish a narrow Down row directly from its ordered K32 sums.
     */
    template <uint8_t CodebookId, bool FixedShortPitch = false>
    __global__ void grouped_native_quantised_down_decode_publish_k32_kernel(
        const int8_t *__restrict__ activation_codes,
        const float *__restrict__ activation_scales,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descriptors,
        const int *__restrict__ expert_ids,
        const float *__restrict__ weights,
        float *__restrict__ routes,
        int N, int K, int num_experts)
    {
        constexpr int kTileN = 64;
        const int column = blockIdx.x * kTileN + threadIdx.x;
        const int route = blockIdx.y;
        if constexpr (FixedShortPitch)
        {
            static_assert(kWideVerifierDownN % kTileN == 0);
            N = kWideVerifierDownN;
            K = kWideVerifierDownK;
        }
        else if (column >= N)
            return;

        const int expert = expert_ids[route];
        const int blocks = K / 32;
        float sum = 0.0f;
        if (expert >= 0 && expert < num_experts)
        {
            const auto *codes = activation_codes + static_cast<size_t>(route) * K;
            const auto *scales = activation_scales + static_cast<size_t>(route) * blocks;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
            if constexpr (FixedShortPitch)
                sum = native_quantised_moe_fixed_down<CodebookId, kWideVerifierDownK, kTileN>(
                    descriptors[expert], column, codes, scales);
            else
#endif
                sum = native_quantised_dot_desc_range_dispatch<CodebookId, NativeMoEDotOrder::DecodeK32>(
                    descriptors[expert], column, codes, scales, N, K, 0, blocks);
        }
        routes[static_cast<size_t>(route) * N + column] =
            weights ? moe_weight_route_rn(weights[route], sum) : sum;
    }

    /** @brief Sum split-K partials while preserving the route dimension. */
    __global__ void grouped_native_quantised_down_kpart_route_reduce_kernel(
        const float *__restrict__ partials,
        const float *__restrict__ route_weights,
        float *__restrict__ route_output,
        int num_active,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int route = blockIdx.y;
        if (route >= num_active || n >= N)
            return;

        const size_t base =
            static_cast<size_t>(route) * static_cast<size_t>(k_partitions) *
            static_cast<size_t>(N);
        float sum = 0.0f;
        for (int k_part = 0; k_part < k_partitions; ++k_part)
        {
            sum = moe_accumulate_rn(
                sum,
                partials[
                    base + static_cast<size_t>(k_part) *
                               static_cast<size_t>(N) +
                    static_cast<size_t>(n)]);
        }
        route_output[
            static_cast<size_t>(route) * static_cast<size_t>(N) +
            static_cast<size_t>(n)] =
            route_weights ? moe_weight_route_rn(route_weights[route], sum) : sum;
    }

    /**
     * @brief Reduce route-major split-K scratch directly to one decode row.
     *
     * This is the non-collective twin of the persistent route publication
     * path.  It first reduces K partitions within one route and then reduces
     * routes in original router order.  The two explicit FP32 loops exactly
     * match `grouped_native_quantised_down_kpart_route_reduce_kernel` followed by
     * `reduce_canonical_route_contributions_kernel`, without materializing the
     * persistent route tensor when no collective needs it.
     */
    __global__ void grouped_native_quantised_down_kpart_routes_reduce_kernel(
        const float *__restrict__ partials,
        const float *__restrict__ route_weights,
        float *__restrict__ output,
        int num_active,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        if (n >= N)
            return;

        float output_sum = 0.0f;
#pragma unroll 1
        for (int route = 0; route < num_active; ++route)
        {
            const size_t route_base =
                static_cast<size_t>(route) *
                static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
            float route_sum = 0.0f;
            for (int k_part = 0; k_part < k_partitions; ++k_part)
            {
                route_sum = moe_accumulate_rn(
                    route_sum,
                    partials[
                        route_base + static_cast<size_t>(k_part) *
                                         static_cast<size_t>(N) +
                        static_cast<size_t>(n)]);
            }
            output_sum = moe_accumulate_rn(
                output_sum,
                moe_weight_route_rn(route_weights[route], route_sum));
        }
        output[n] = output_sum;
    }


    /** @brief Load one contiguous expert weight using the published format tag. */
    __device__ __forceinline__ float load_floating_moe_weight(
        const void *__restrict__ data,
        size_t index,
        llaminar2::DeviceMoEWeightFormat format)
    {
        if (format == llaminar2::DeviceMoEWeightFormat::FP32)
            return static_cast<const float *>(data)[index];
        if (format == llaminar2::DeviceMoEWeightFormat::BF16)
        {
            return __bfloat162float(
                static_cast<const __nv_bfloat16 *>(data)[index]);
        }
        return __half2float(static_cast<const __half *>(data)[index]);
    }

    /**
     * @brief Fixed-tree gate/up decode for a device-selected floating expert.
     *
     * One block owns one `(route, output-column)` pair. All weight formats
     * convert inside the same strided K loop and reduce through the same
     * 256-thread tree, so M=1 decode and grouped verifier rows have an
     * identical arithmetic schedule independent of BLAS heuristics.
     */
    __global__ __launch_bounds__(kThreads)
    void grouped_floating_gate_up_decode_kernel(
        const float *__restrict__ hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ gate_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_ids,
        float *const *__restrict__ gate_outputs,
        float *const *__restrict__ up_outputs,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format)
    {
        __shared__ float gate_sums[kThreads];
        __shared__ float up_sums[kThreads];
        const int n = static_cast<int>(blockIdx.x);
        const int route = logical_row_from_grid_yz();
        if (route >= num_active || n >= intermediate)
            return;

        const int expert = expert_ids[route];
        const bool valid = expert >= 0 && expert < num_experts &&
                           gate_outputs[route] && up_outputs[route];
        llaminar2::DeviceMoEFloatingMatrixDesc gate{};
        llaminar2::DeviceMoEFloatingMatrixDesc up{};
        if (valid)
        {
            gate = gate_descs[expert];
            up = up_descs[expert];
        }
        const bool ready = valid && gate.data && up.data &&
                           gate.n == intermediate && gate.k == d_model &&
                           up.n == intermediate && up.k == d_model;

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        if (ready)
        {
            const size_t weight_base =
                static_cast<size_t>(n) * static_cast<size_t>(d_model);
            for (int k = static_cast<int>(threadIdx.x);
                 k < d_model;
                 k += kThreads)
            {
                const float activation = hidden[k];
                gate_sum = fmaf(
                    activation,
                    load_floating_moe_weight(
                        gate.data, weight_base + static_cast<size_t>(k), format),
                    gate_sum);
                up_sum = fmaf(
                    activation,
                    load_floating_moe_weight(
                        up.data, weight_base + static_cast<size_t>(k), format),
                    up_sum);
            }
        }
        gate_sums[threadIdx.x] = gate_sum;
        up_sums[threadIdx.x] = up_sum;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1)
        {
            if (static_cast<int>(threadIdx.x) < stride)
            {
                gate_sums[threadIdx.x] = moe_accumulate_rn(
                    gate_sums[threadIdx.x], gate_sums[threadIdx.x + stride]);
                up_sums[threadIdx.x] = moe_accumulate_rn(
                    up_sums[threadIdx.x], up_sums[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && valid)
        {
            gate_outputs[route][n] = ready ? gate_sums[0] : 0.0f;
            up_outputs[route][n] = ready ? up_sums[0] : 0.0f;
        }
    }

    /**
     * @brief Fixed-tree floating SwiGLU/down decode with canonical route order.
     *
     * Direct output uses one block per output column and visits routes in
     * router order. Canonical publication uses one block per route/column and
     * writes the already-rounded weighted contribution. Both forms therefore
     * share the same per-route dot tree and explicit weighting boundary.
     */
    __global__ __launch_bounds__(kThreads)
    void grouped_floating_swiglu_down_decode_kernel(
        const float *const *__restrict__ gate_rows,
        const float *const *__restrict__ up_rows,
        const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ down_descs,
        const int *__restrict__ expert_ids,
        const float *__restrict__ route_weights,
        float *__restrict__ output,
        float *__restrict__ canonical_routes,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format)
    {
        __shared__ float sums[kThreads];
        const int n = static_cast<int>(blockIdx.x);
        const int publication_row = logical_row_from_grid_yz();
        const bool canonical = canonical_routes != nullptr;
        if (n >= d_model || (canonical && publication_row >= num_active))
            return;

        const int route_begin = canonical ? publication_row : 0;
        const int route_end = canonical ? publication_row + 1 : num_active;
        float output_sum = 0.0f;
        for (int route = route_begin; route < route_end; ++route)
        {
            const int expert = expert_ids[route];
            const bool valid = expert >= 0 && expert < num_experts &&
                               gate_rows[route] && up_rows[route];
            llaminar2::DeviceMoEFloatingMatrixDesc down{};
            if (valid)
                down = down_descs[expert];
            const bool ready = valid && down.data &&
                               down.n == d_model && down.k == intermediate;
            float sum = 0.0f;
            if (ready)
            {
                const size_t weight_base =
                    static_cast<size_t>(n) * static_cast<size_t>(intermediate);
                for (int k = static_cast<int>(threadIdx.x);
                     k < intermediate;
                     k += kThreads)
                {
                    const float gate = gate_rows[route][k];
                    const float up = up_rows[route][k];
                    const float swiglu =
                        llaminar2::device_swiglu_contract::swigluValue(
                            gate, up);
                    sum = fmaf(
                        swiglu,
                        load_floating_moe_weight(
                            down.data,
                            weight_base + static_cast<size_t>(k),
                            format),
                        sum);
                }
            }
            sums[threadIdx.x] = sum;
            __syncthreads();
            for (int stride = kThreads / 2; stride > 0; stride >>= 1)
            {
                if (static_cast<int>(threadIdx.x) < stride)
                {
                    sums[threadIdx.x] = moe_accumulate_rn(
                        sums[threadIdx.x], sums[threadIdx.x + stride]);
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                const float projected = ready ? sums[0] : 0.0f;
                output_sum = route_weights
                    ? moe_accumulate_rn(output_sum, ready
                        ? moe_weight_route_rn(route_weights[route], projected) : 0.0f)
                    : projected;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            if (canonical)
            {
                canonical_routes[
                    static_cast<size_t>(publication_row) *
                        static_cast<size_t>(d_model) +
                    static_cast<size_t>(n)] = output_sum;
            }
            else
            {
                output[n] = output_sum;
            }
        }
    }

    /**
     * @brief Project eight floating columns per block with the exact serial tree.
     *
     * Each width-32 subgroup owns one column; its eight register accumulators
     * retain the 256 logical K partitions. Whole-subgroup tails and absent routes
     * are safe because no block barrier or cross-column state is used.
     */
    __global__ __launch_bounds__(llaminar2::device_moe_floating_warp::threads)
    void grouped_floating_gate_up_swiglu_prefill_kernel(
        const float *__restrict__ hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ gate_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        float *__restrict__ grouped_swiglu,
        int total_slots,
        int top_k,
        int intermediate,
        int d_model,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format)
    {
        namespace warp = llaminar2::device_moe_floating_warp;
        const int lane = threadIdx.x % warp::width;
        const int n = blockIdx.x * warp::columns_per_block + threadIdx.x / warp::width;
        const int original_slot = logical_row_from_grid_yz();
        if (original_slot >= total_slots || n >= intermediate)
            return;
        const int grouped_slot = original_to_grouped[original_slot];
        const int expert = original_expert_ids ? original_expert_ids[original_slot] : 0;
        if (grouped_slot < 0 || expert < 0 || expert >= num_experts)
            return;

        const auto gate = gate_descs[expert];
        const auto up = up_descs[expert];
        const bool ready = gate.data && up.data &&
                           gate.n == intermediate && gate.k == d_model &&
                           up.n == intermediate && up.k == d_model;
        float gate_sums[warp::partitions_per_lane]{};
        float up_sums[warp::partitions_per_lane]{};
        if (ready)
        {
            const float *row = hidden + static_cast<size_t>(original_slot / top_k) * d_model;
            const size_t weight_base = static_cast<size_t>(n) * d_model;
            // Register p emulates original thread lane+32*p. Moving between
            // columns changes scheduling, never that thread's FMA order.
            for (int base = 0; base < d_model; base += warp::partitions)
            {
    #pragma unroll
                for (int partition = 0; partition < warp::partitions_per_lane; ++partition)
                {
                    const int k = base + lane + partition * warp::width;
                    if (k < d_model)
                    {
                        const float activation = row[k];
                        gate_sums[partition] = fmaf(activation,
                            load_floating_moe_weight(gate.data, weight_base + k, format), gate_sums[partition]);
                        up_sums[partition] = fmaf(activation,
                            load_floating_moe_weight(up.data, weight_base + k, format), up_sums[partition]);
                    }
                }
            }
        }
        const float gate_sum = warp::reduce(gate_sums);
        const float up_sum = warp::reduce(up_sums);
        if (lane == 0)
        {
            const size_t index = static_cast<size_t>(grouped_slot) * intermediate + n;
            grouped_swiglu[index] = ready ? llaminar2::device_swiglu_contract::swigluValue(gate_sum, up_sum) : 0.0f;
        }
    }

    /**
     * @brief Down-project eight columns per block without shared reduction scratch.
     *
     * The preceding node already evaluated SwiGLU once. Each subgroup reads those
     * FP32 bits, emulates the serial 256-partition FMA/tree, and publishes either
     * one canonical route or the original ordered route sum. Missing routes still
     * contribute an explicit rounded +0, including signed-zero semantics.
     */
    __global__ __launch_bounds__(llaminar2::device_moe_floating_warp::threads)
    void grouped_floating_down_prefill_kernel(
        const float *__restrict__ grouped_swiglu,
        const llaminar2::DeviceMoEFloatingMatrixDesc *__restrict__ down_descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        const float *__restrict__ grouped_weights,
        float *__restrict__ output,
        float *__restrict__ canonical_routes,
        int seq_len,
        int total_slots,
        int top_k,
        int d_model,
        int intermediate,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format)
    {
        namespace warp = llaminar2::device_moe_floating_warp;
        const int lane = threadIdx.x % warp::width;
        const int n = blockIdx.x * warp::columns_per_block + threadIdx.x / warp::width;
        const int publication_row = logical_row_from_grid_yz();
        const bool canonical = canonical_routes != nullptr;
        if (n >= d_model || (canonical ? publication_row >= total_slots : publication_row >= seq_len))
            return;

        const int route_begin = canonical ? publication_row : publication_row * top_k;
        const int route_end = canonical ? publication_row + 1 : route_begin + top_k;
        float output_sum = 0.0f;
        for (int original_slot = route_begin; original_slot < route_end; ++original_slot)
        {
            const int grouped_slot = original_to_grouped[original_slot];
            const int expert = original_expert_ids ? original_expert_ids[original_slot] : 0;
            const bool valid = grouped_slot >= 0 && expert >= 0 && expert < num_experts;
            llaminar2::DeviceMoEFloatingMatrixDesc down{};
            if (valid)
                down = down_descs[expert];
            const bool ready = valid && down.data && down.n == d_model && down.k == intermediate;
            if (!ready)
            {
                if (lane == 0)
                    output_sum = moe_accumulate_rn(output_sum, 0.0f);
                continue;
            }
            float sums[warp::partitions_per_lane]{};
            const size_t row_base = static_cast<size_t>(grouped_slot) * intermediate;
            const size_t weight_base = static_cast<size_t>(n) * intermediate;
            for (int base = 0; base < intermediate; base += warp::partitions)
            {
    #pragma unroll
                for (int partition = 0; partition < warp::partitions_per_lane; ++partition)
                {
                    const int k = base + lane + partition * warp::width;
                    if (k < intermediate)
                        sums[partition] = fmaf(grouped_swiglu[row_base + k],
                            load_floating_moe_weight(down.data, weight_base + k, format), sums[partition]);
                }
            }
            const float sum = warp::reduce(sums);
            if (lane == 0)
                output_sum = grouped_weights
                    ? moe_accumulate_rn(output_sum, moe_weight_route_rn(grouped_weights[grouped_slot], sum))
                    : sum;
        }
        if (lane == 0)
        {
            const size_t index = static_cast<size_t>(publication_row) * d_model + n;
            if (canonical)
                canonical_routes[index] = output_sum;
            else
                output[index] = output_sum;
        }
    }

    /**
     * @brief Publish one shared partial into a rank-addressed canonical bank.
     *
     * Every launch overwrites the complete participant-bank suffix. A bank row
     * owned by this participant copies the local shared FFN output; all peer
     * banks and padded rows receive exact zero. Route slots precede the suffix
     * and are intentionally untouched because the routed expert kernel owns them.
     */
    __global__ void publish_shared_expert_rank_bank_kernel(
        const float *__restrict__ shared_output,
        float *__restrict__ canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *__restrict__ device_effective_seq_len)
    {
        const int bank_row = logical_row_from_grid_yz();
        const int participant = bank_row / seq_len;
        const int token = bank_row - participant * seq_len;
        if (participant >= participant_count || token >= seq_len)
            return;

        const int effective_seq_len =
            clamp_effective_seq_len(seq_len, device_effective_seq_len);
        const bool copy_local =
            participant == participant_index && token < effective_seq_len;
        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t route_elements =
            row_elements * static_cast<size_t>(top_k);
        const size_t source_offset =
            static_cast<size_t>(token) * static_cast<size_t>(d_model);
        const size_t destination_offset =
            route_elements +
            static_cast<size_t>(participant) * row_elements + source_offset;

        if ((d_model & 3) == 0)
        {
            const int n4 = d_model >> 2;
            const float4 *source4 =
                reinterpret_cast<const float4 *>(shared_output + source_offset);
            float4 *destination4 =
                reinterpret_cast<float4 *>(
                    canonical_publication + destination_offset);
            // copy_local is uniform for this bank row. Separate value stores
            // prevent an lvalue conditional from materializing zero in private
            // memory and selecting its address for every peer-bank vector.
            if (copy_local)
            {
                for (int index = blockIdx.x * blockDim.x + threadIdx.x;
                     index < n4; index += gridDim.x * blockDim.x)
                    destination4[index] = source4[index];
            }
            else
            {
                for (int index = blockIdx.x * blockDim.x + threadIdx.x;
                     index < n4; index += gridDim.x * blockDim.x)
                    destination4[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            return;
        }

        /*
         * A ragged row gives every token after row zero a potentially
         * non-16-byte-aligned base. Reinterpreting that base as float4 is not
         * a legal vector access, so width totality requires a scalar row path.
         */
        for (int column = blockIdx.x * blockDim.x + threadIdx.x;
             column < d_model;
             column += gridDim.x * blockDim.x)
        {
            canonical_publication[destination_offset + column] =
                copy_local ? shared_output[source_offset + column] : 0.0f;
        }
    }

    /**
     * @brief Fold routed slots and shared rank banks into one final MoE row.
     *
     * The thread geometry and gate-dot reduction are deliberately identical to
     * shared_expert_gate_add_kernel. The two-dimensional launch assigns one
     * block to each token/output-column tile. Every tile recomputes the same
     * fixed-order gate value, so no inter-block publication or second kernel is
     * required and every tile observes a bit-identical gate. Each output column
     * then walks router slots and participant banks in canonical increasing
     * order. Explicit FP32 round-to-nearest intrinsics preserve the global-memory
     * rounding boundaries of the former multi-kernel pipeline while avoiding
     * its extra launches.
     */
    __global__ void finalize_canonical_moe_publication_kernel(
        const float *__restrict__ input,
        const float *__restrict__ gate_inp,
        const float *__restrict__ canonical_publication,
        float *__restrict__ routed_output,
        float *__restrict__ shared_output,
        float *__restrict__ combined_output,
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        const int *__restrict__ device_effective_seq_len)
    {
        const int output_tile = static_cast<int>(blockIdx.x);
        const int token = logical_row_from_grid_yz();
        const int first_column =
            output_tile * kCanonicalMoEOutputTileColumns;
        const int unbounded_last_column =
            first_column + kCanonicalMoEOutputTileColumns;
        const int last_column =
            unbounded_last_column < d_model
                ? unbounded_last_column
                : d_model;
        if (token >= seq_len || first_column >= d_model)
            return;

        __shared__ float partial[kThreads];
        const size_t row_offset =
            static_cast<size_t>(token) * static_cast<size_t>(d_model);
        const int effective_seq_len =
            clamp_effective_seq_len(seq_len, device_effective_seq_len);
        if (token >= effective_seq_len)
        {
            if ((d_model & 3) == 0)
            {
                const int first4 = first_column >> 2;
                const int last4 = last_column >> 2;
                const float4 zero =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 *routed4 =
                    reinterpret_cast<float4 *>(routed_output + row_offset);
                float4 *shared4 =
                    reinterpret_cast<float4 *>(shared_output + row_offset);
                float4 *combined4 =
                    reinterpret_cast<float4 *>(combined_output + row_offset);
                for (int index = first4 + threadIdx.x;
                     index < last4;
                     index += blockDim.x)
                {
                    routed4[index] = zero;
                    shared4[index] = zero;
                    combined4[index] = zero;
                }
            }
            else
            {
                for (int column = first_column + threadIdx.x;
                     column < last_column;
                     column += blockDim.x)
                {
                    routed_output[row_offset + column] = 0.0f;
                    shared_output[row_offset + column] = 0.0f;
                    combined_output[row_offset + column] = 0.0f;
                }
            }
            return;
        }

        float dot = 0.0f;
        if ((d_model & 3) == 0)
        {
            const int n4 = d_model >> 2;
            const float4 *input4 =
                reinterpret_cast<const float4 *>(input + row_offset);
            const float4 *gate4 =
                reinterpret_cast<const float4 *>(gate_inp);
            for (int index = threadIdx.x;
                 index < n4;
                 index += blockDim.x)
            {
                const float4 x = input4[index];
                const float4 g = gate4[index];
                dot += x.x * g.x + x.y * g.y + x.z * g.z + x.w * g.w;
            }
        }
        else
        {
            for (int column = threadIdx.x;
                 column < d_model;
                 column += blockDim.x)
            {
                dot += input[row_offset + column] * gate_inp[column];
            }
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float gate = 1.0f / (1.0f + expf(-partial[0]));

        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t route_elements =
            row_elements * static_cast<size_t>(top_k);
        const float *rank_banks = canonical_publication + route_elements;
        if ((d_model & 3) == 0)
        {
            const int first4 = first_column >> 2;
            const int last4 = last_column >> 2;
            float4 *routed4 =
                reinterpret_cast<float4 *>(routed_output + row_offset);
            float4 *shared4 =
                reinterpret_cast<float4 *>(shared_output + row_offset);
            float4 *combined4 =
                reinterpret_cast<float4 *>(combined_output + row_offset);

            for (int index = first4 + threadIdx.x;
                 index < last4;
                 index += blockDim.x)
            {
                float4 routed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 1
                for (int route = 0; route < top_k; ++route)
                {
                    const size_t route_row =
                        (static_cast<size_t>(token) *
                             static_cast<size_t>(top_k) +
                         static_cast<size_t>(route)) *
                        static_cast<size_t>(d_model);
                    const float4 contribution =
                        reinterpret_cast<const float4 *>(
                            canonical_publication + route_row)[index];
                    routed.x =
                        moe_accumulate_rn(routed.x, contribution.x);
                    routed.y =
                        moe_accumulate_rn(routed.y, contribution.y);
                    routed.z =
                        moe_accumulate_rn(routed.z, contribution.z);
                    routed.w =
                        moe_accumulate_rn(routed.w, contribution.w);
                }

                float4 shared = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 1
                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    const size_t bank_row =
                        static_cast<size_t>(participant) * row_elements +
                        row_offset;
                    const float4 contribution =
                        reinterpret_cast<const float4 *>(
                            rank_banks + bank_row)[index];
                    shared.x =
                        moe_accumulate_rn(shared.x, contribution.x);
                    shared.y =
                        moe_accumulate_rn(shared.y, contribution.y);
                    shared.z =
                        moe_accumulate_rn(shared.z, contribution.z);
                    shared.w =
                        moe_accumulate_rn(shared.w, contribution.w);
                }

                const float4 gated = make_float4(
                    __fmul_rn(gate, shared.x),
                    __fmul_rn(gate, shared.y),
                    __fmul_rn(gate, shared.z),
                    __fmul_rn(gate, shared.w));
                routed4[index] = routed;
                shared4[index] = gated;
                combined4[index] = make_float4(
                    __fadd_rn(routed.x, gated.x),
                    __fadd_rn(routed.y, gated.y),
                    __fadd_rn(routed.z, gated.z),
                    __fadd_rn(routed.w, gated.w));
            }
            return;
        }

        for (int column = first_column + threadIdx.x;
             column < last_column;
             column += blockDim.x)
        {
            float routed = 0.0f;
#pragma unroll 1
            for (int route = 0; route < top_k; ++route)
            {
                const size_t route_index =
                    (static_cast<size_t>(token) *
                         static_cast<size_t>(top_k) +
                     static_cast<size_t>(route)) *
                        static_cast<size_t>(d_model) +
                    static_cast<size_t>(column);
                routed = moe_accumulate_rn(
                    routed, canonical_publication[route_index]);
            }
            float shared = 0.0f;
#pragma unroll 1
            for (int participant = 0;
                 participant < participant_count;
                 ++participant)
            {
                const size_t shared_index =
                    static_cast<size_t>(participant) * row_elements +
                    row_offset + static_cast<size_t>(column);
                shared = moe_accumulate_rn(shared, rank_banks[shared_index]);
            }
            const size_t output_index =
                row_offset + static_cast<size_t>(column);
            const float gated = __fmul_rn(gate, shared);
            routed_output[output_index] = routed;
            shared_output[output_index] = gated;
            combined_output[output_index] = __fadd_rn(routed, gated);
        }
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_quantised_gate_up_kpart_decode_runtime_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int slot = blockIdx.z;
        if (slot >= num_active || k_part >= k_partitions || n >= N)
            return;

        const size_t partial_index =
            (static_cast<size_t>(slot) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);

        const int expert_id = expert_ids[slot];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (execution_bank > 1u)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;
        if (b_start >= b_end)
        {
            gate_partials[partial_index] = 0.0f;
            up_partials[partial_index] = 0.0f;
            return;
        }

        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const DeviceNativeQuantisedMatrixDesc gate_desc = bank.experts[expert_id].gate;
        const DeviceNativeQuantisedMatrixDesc up_desc = bank.experts[expert_id].up;
        gate_partials[partial_index] = native_quantised_dot_desc_range_dispatch<CodebookId>(
            gate_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        up_partials[partial_index] = native_quantised_dot_desc_range_dispatch<CodebookId>(
            up_desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
    }

    template <uint8_t CodebookId>
    __global__ void grouped_native_quantised_down_kpart_decode_runtime_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceMoELayerRuntimeView *__restrict__ runtime,
        const int *__restrict__ expert_ids,
        float *__restrict__ partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int k_part = blockIdx.y;
        const int route = blockIdx.z;
        if (route >= num_active || k_part >= k_partitions || n >= N)
            return;

        const size_t partial_index =
            (static_cast<size_t>(route) *
                 static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);

        const int blocks_per_row = K / 32;
        const int blocks_per_part = (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if (b_end > blocks_per_row)
            b_end = blocks_per_row;
        if (b_start >= b_end)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        const int expert_id = expert_ids[route];
        if (expert_id < 0 || expert_id >= num_experts)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        const uint32_t execution_bank = runtime_execution_bank(runtime);
        if (execution_bank > 1u)
        {
            partials[partial_index] = 0.0f;
            return;
        }

        const DeviceMoEPlacementBankView &bank =
            runtime_placement_banks(runtime)[execution_bank];
        const DeviceNativeQuantisedMatrixDesc desc = bank.experts[expert_id].down;
        const int8_t *route_A =
            A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
        const float *route_scales =
            scales_A_blockwise +
            static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
        partials[partial_index] =
            native_quantised_dot_desc_range_dispatch<CodebookId>(
                desc, n, route_A, route_scales, N, K, b_start, b_end);
    }


    /**
     * @brief Fold canonical routes with overlapping loads and serial arithmetic.
     *
     * A bounded register window exposes independent memory requests. It does
     * not change the router-order addition tree or add persistent GPU storage.
     */
    __global__ void reduce_canonical_route_contributions_kernel(
        const float *__restrict__ route_contributions,
        float *__restrict__ output,
        int seq_len,
        int top_k,
        int d_model)
    {
        constexpr int kTileN = 256;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token = logical_row_from_grid_yz();
        if (token >= seq_len || n >= d_model)
            return;

        const size_t first = static_cast<size_t>(token) * top_k * d_model + n;
        const float sum = llaminar2::device_ordered_route_fold::sum(
            route_contributions + first, static_cast<size_t>(d_model), top_k);
        output[
            static_cast<size_t>(token) * static_cast<size_t>(d_model) +
            static_cast<size_t>(n)] = sum;
    }

} // namespace

extern "C"
{

    /** @brief Launch fixed-tree floating gate/up decode. */
    bool cudaMoE_grouped_gate_up_floating_decode_table(
        const float *d_hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_gate_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_up_descs,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int num_active,
        int intermediate,
        int d_model,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream)
    {
        if (!d_hidden || !d_gate_descs || !d_up_descs || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || num_active <= 0 ||
            intermediate <= 0 || d_model <= 0 || num_experts <= 0 ||
            !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
        {
            return false;
        }
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        grouped_floating_gate_up_decode_kernel<<<
            make_row_major_grid(
                static_cast<unsigned int>(intermediate), num_active),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
                d_hidden,
                d_gate_descs,
                d_up_descs,
                d_expert_ids,
                d_gate_outputs,
                d_up_outputs,
                num_active,
                intermediate,
                d_model,
                num_experts,
                format);
        return finishLaunch("cudaMoE_grouped_gate_up_floating_decode_table");
    }

    /** @brief Launch fixed-tree floating SwiGLU/down decode. */
    bool cudaMoE_grouped_swiglu_down_floating_decode_table(
        const float *const *d_gate_rows,
        const float *const *d_up_rows,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_down_descs,
        const int *d_expert_ids,
        const float *d_route_weights,
        float *d_output,
        float *d_canonical_routes,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream)
    {
        if (!d_gate_rows || !d_up_rows || !d_down_descs || !d_expert_ids ||
            (!d_route_weights && (d_output || !d_canonical_routes)) ||
            (!d_output && !d_canonical_routes) ||
            num_active <= 0 || d_model <= 0 || intermediate <= 0 ||
            num_experts <= 0 ||
            !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
        {
            return false;
        }
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        const int publication_rows = d_canonical_routes ? num_active : 1;
        grouped_floating_swiglu_down_decode_kernel<<<
            make_row_major_grid(
                static_cast<unsigned int>(d_model), publication_rows),
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
                d_gate_rows,
                d_up_rows,
                d_down_descs,
                d_expert_ids,
                d_route_weights,
                d_output,
                d_canonical_routes,
                num_active,
                d_model,
                intermediate,
                num_experts,
                format);
        return finishLaunch(
            "cudaMoE_grouped_swiglu_down_floating_decode_table");
    }

    /**
     * @brief Launch the declared fixed-tree floating phase on the exact stream.
     * @param d_grouped_swiglu Existing exclusive arena FP32 intermediate.
     * @param execution Complete transaction, gate/up producer, or full-K down-column consumer.
     *
     * Complete execution retains two captured nodes. A split phase borrows the
     * same storage and serial arithmetic; it adds no conversion or allocation.
     */
    bool cudaMoE_grouped_floating_prefill_pipeline(
        const float *d_hidden,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_gate_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_up_descs,
        const llaminar2::DeviceMoEFloatingMatrixDesc *d_down_descs,
        const int *d_original_to_grouped,
        const int *d_original_expert_ids,
        const float *d_grouped_weights,
        float *d_grouped_swiglu,
        float *d_output,
        float *d_canonical_routes,
        int seq_len,
        int total_slots,
        int top_k,
        int d_model,
        int intermediate,
        int num_experts,
        llaminar2::DeviceMoEWeightFormat format,
        int device_idx,
        void *stream,
        const llaminar2::MoEPrefillProjectionExecution &execution)
    {
        const bool raw_cohort =
            execution.routeWeighting() == llaminar2::MoEExpertRouteWeighting::CompletedCohort;
        if ((raw_cohort && (d_output || !d_canonical_routes)))
            return false;
        if ((execution.executesGateUp() && (!d_hidden || !d_gate_descs || !d_up_descs)) ||
            (execution.executesDown() && (!d_down_descs || !d_grouped_weights ||
                (!d_output && !d_canonical_routes))) ||
            !d_original_to_grouped ||
            (!d_original_expert_ids && num_experts != 1) ||
            !d_grouped_swiglu || seq_len <= 0 || execution.modelColumns() != d_model ||
            total_slots != static_cast<std::int64_t>(seq_len) * top_k || top_k <= 0 || d_model <= 0 ||
            intermediate <= 0 || num_experts <= 0 ||
            !llaminar2::deviceMoEWeightFormatIsFloating(format) || !stream)
        {
            return false;
        }
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        // Only independent output columns are partitioned; K's fixed tree is unchanged.
        if (execution.executesGateUp())
        {
            grouped_floating_gate_up_swiglu_prefill_kernel<<<
                make_row_major_grid(
                    static_cast<unsigned int>((intermediate + llaminar2::device_moe_floating_warp::columns_per_block - 1) /
                        llaminar2::device_moe_floating_warp::columns_per_block), total_slots),
                kThreads,
                0,
                cuda_stream>>>(
                    d_hidden,
                    d_gate_descs,
                    d_up_descs,
                    d_original_to_grouped,
                    d_original_expert_ids,
                    d_grouped_swiglu,
                    total_slots,
                    top_k,
                    intermediate,
                    d_model,
                    num_experts,
                    format);
            if (!finishLaunch("cudaMoE_grouped_floating_prefill_gate_up"))
                return false;
        }
        if (!execution.executesDown())
            return true;
        const int down_columns = execution.columnCount();
        const int publication_rows = d_canonical_routes ? total_slots : seq_len;
        grouped_floating_down_prefill_kernel<<<
            make_row_major_grid(
                static_cast<unsigned int>((down_columns + llaminar2::device_moe_floating_warp::columns_per_block - 1) /
                    llaminar2::device_moe_floating_warp::columns_per_block), publication_rows),
            kThreads,
            0,
            cuda_stream>>>(
                d_grouped_swiglu,
                d_down_descs,
                d_original_to_grouped,
                d_original_expert_ids,
                raw_cohort ? nullptr : d_grouped_weights,
                d_output,
                d_canonical_routes,
                seq_len,
                total_slots,
                top_k,
                down_columns,
                intermediate,
                num_experts,
                format);
        return finishLaunch("cudaMoE_grouped_floating_prefill_down");
    }

    bool cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart(
        const float *d_hidden,
        const DeviceNativeQuantisedMatrixDesc *d_gate_desc_table,
        const DeviceNativeQuantisedMatrixDesc *d_up_desc_table,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream,
        llaminar2::CUDAMoESwiGLUPublication swiglu)
    {
        const bool valid_k_partitions =
            k_partitions ==
            llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        const bool publish_swiglu = swiglu.codes && swiglu.scales;
        const bool valid_output = publish_swiglu
            ? !d_gate_outputs && !d_up_outputs && N % 32 == 0
            : d_gate_outputs && d_up_outputs && !swiglu.codes && !swiglu.scales;
        if (!stream || !d_hidden || !d_gate_desc_table || !d_up_desc_table || !d_expert_ids ||
            !valid_output || !d_hidden_int8 || !d_hidden_scales ||
            !d_gate_partials || !d_up_partials ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
            (K % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        // Step 1: blockwise-quantize the shared hidden activation row once.
        if (!hidden_prequantized)
        {
            const int blocks_per_row = K / 32;
            if (!cudaMoE_internal_quantize_hidden_blockwise(d_hidden, d_hidden_int8, d_hidden_scales, K, cuda_stream)) return false;
            if (!finishLaunch("cudaMoE_grouped_gate_up_kpart_hidden_quantize"))
                return false;
        }

        // Step 2: split-K scatter — each (n-tile, k-partition, expert-slot) block
        // reduces its K-slice into the partials buffer.
        constexpr int kTileN = 64;
        dim3 partial_grid((N + kTileN - 1) / kTileN, k_partitions, num_active);
        dim3 block(kTileN);

        const bool fixed_qfn = [&]() {
            if (!publish_swiglu || N != kQFNDecodeGateUpN || K != kQFNDecodeGateUpK)
                return false;
            return is_sm86_cuda_device(device_idx);
        }();
        const auto launch_gate = [&]<uint8_t CodebookId>() {
            const auto issue = [&]<bool FixedQFN>() {
                grouped_native_quantised_gate_up_kpart_decode_kernel<CodebookId, FixedQFN>
                    <<<partial_grid, block, 0, cuda_stream>>>(
                        d_hidden_int8, d_hidden_scales, d_gate_desc_table, d_up_desc_table, d_expert_ids,
                        d_gate_partials, d_up_partials, num_active, N, K, num_experts, k_partitions);
            };
            if constexpr (kQFNDecodeFixedGateUpCodebook<CodebookId>)
            {
                if (fixed_qfn)
                    return issue.template operator()<true>();
            }
            issue.template operator()<false>();
        };

#define LAUNCH_GROUPED_GATE_UP_KPART(CB) launch_gate.template operator()<CB>()

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP_KPART(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP_KPART(4); break;
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook: LAUNCH_GROUPED_GATE_UP_KPART(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook); break;
        case 5:  LAUNCH_GROUPED_GATE_UP_KPART(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP_KPART(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP_KPART(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP_KPART(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP_KPART(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP_KPART(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP_KPART(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP_KPART(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP_KPART(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP_KPART(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP_KPART(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP_KPART(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP_KPART(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP_KPART(19); break;
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook: LAUNCH_GROUPED_GATE_UP_KPART(llaminar2::kNativeQuantisedExpandedInt8MinCodebook); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_GATE_UP_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP_KPART

        if (!finishLaunch("cudaMoE_grouped_gate_up_kpart_partial"))
            return false;

        // The original FP32 result is published only for callers that consume
        // gate/up separately. The fused path gives Down its canonical Q8 planes.
        dim3 reduce_grid((N + kTileN - 1) / kTileN, num_active);
        if (publish_swiglu)
        {
            grouped_native_quantised_gate_up_kpart_reduce_kernel<true><<<reduce_grid, block, 0, cuda_stream>>>(
                d_gate_partials, d_up_partials, d_expert_ids, nullptr, nullptr,
                num_active, N, k_partitions, swiglu);
        }
        else
        {
            grouped_native_quantised_gate_up_kpart_reduce_kernel<false><<<reduce_grid, block, 0, cuda_stream>>>(
                d_gate_partials, d_up_partials, d_expert_ids, d_gate_outputs, d_up_outputs,
                num_active, N, k_partitions, {});
        }

        return finishLaunch("cudaMoE_grouped_gate_up_native_quantised_decode_table_kpart");
    }

    bool cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart(
        const float *d_hidden,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        float *const *d_gate_outputs,
        float *const *d_up_outputs,
        int8_t *d_hidden_int8,
        float *d_hidden_scales,
        bool hidden_prequantized,
        float *d_gate_partials,
        float *d_up_partials,
        int num_active,
        int N,
        int K,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions =
            k_partitions ==
            llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        if (!d_hidden || !d_runtime_layer || !d_expert_ids ||
            !d_gate_outputs || !d_up_outputs || !d_hidden_int8 || !d_hidden_scales ||
            !d_gate_partials || !d_up_partials ||
            num_active <= 0 || N <= 0 || K <= 0 || num_experts <= 0 ||
            (K % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        if (!hidden_prequantized)
        {
            const int blocks_per_row = K / 32;
            if (!cudaMoE_internal_quantize_hidden_blockwise(d_hidden, d_hidden_int8, d_hidden_scales, K, cuda_stream)) return false;
            if (!finishLaunch("cudaMoE_grouped_gate_up_runtime_kpart_hidden_quantize"))
                return false;
        }

        constexpr int kTileN = 64;
        dim3 partial_grid((N + kTileN - 1) / kTileN, k_partitions, num_active);
        dim3 block(kTileN);
        const auto *runtime = static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer);

#define LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(CB)                                                   \
    grouped_native_quantised_gate_up_kpart_decode_runtime_kernel<CB><<<partial_grid, block, 0, cuda_stream>>>( \
        d_hidden_int8, d_hidden_scales, runtime, d_expert_ids,                                      \
        d_gate_partials, d_up_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(0); break;
        case 4:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(4); break;
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook); break;
        case 5:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(5); break;
        case 6:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(6); break;
        case 7:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(7); break;
        case 8:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(8); break;
        case 9:  LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(9); break;
        case 10: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(10); break;
        case 11: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(11); break;
        case 12: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(12); break;
        case 13: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(13); break;
        case 14: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(14); break;
        case 15: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(15); break;
        case 16: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(16); break;
        case 17: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(17); break;
        case 19: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(19); break;
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(llaminar2::kNativeQuantisedExpandedInt8MinCodebook); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_GATE_UP_RUNTIME_KPART

        if (!finishLaunch("cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart partial"))
            return false;

        dim3 reduce_grid((N + kTileN - 1) / kTileN, num_active);
        grouped_native_quantised_gate_up_kpart_reduce_kernel<false><<<reduce_grid, block, 0, cuda_stream>>>(
            d_gate_partials, d_up_partials, d_expert_ids, d_gate_outputs, d_up_outputs,
            num_active, N, k_partitions, {});

        return finishLaunch("cudaMoE_grouped_gate_up_native_quantised_decode_runtime_kpart reduce");
    }

    bool cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const DeviceNativeQuantisedMatrixDesc *d_desc_table,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        float *d_canonical_route_contributions,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream,
        llaminar2::CUDAMoEDecodeDownInput input)
    {
        const bool valid_k_partitions =
            k_partitions ==
            llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions;
        using DownInput = llaminar2::CUDAMoEDecodeDownInput;
        const bool valid_input =
            (input == DownInput::PublishedSwiGLU && !d_gate_ptrs && !d_up_ptrs) ||
            (input == DownInput::FloatGateUp && d_gate_ptrs && d_up_ptrs);
        const bool direct_route_publication = input == DownInput::PublishedSwiGLU &&
            d_canonical_route_contributions && !d_output && intermediate / 32 <= k_partitions;
        if (!stream || !valid_input || !d_desc_table || !d_expert_ids ||
            (!d_weights && (d_output || !d_canonical_route_contributions)) ||
            !d_swiglu_int8 || !d_swiglu_scales || (!direct_route_publication && !d_down_partials) ||
            (!d_output && !d_canonical_route_contributions) ||
            num_active <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 ||
            (intermediate % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        if (input == DownInput::FloatGateUp)
        {
            grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
                d_gate_ptrs, d_up_ptrs, d_expert_ids, d_swiglu_int8, d_swiglu_scales, num_active, intermediate);
            if (!finishLaunch("cudaMoE_grouped_swiglu_quantize"))
                return false;
        }

        constexpr int kTileN = 64;
        const int N = d_model;
        const int K = intermediate;
        const bool canonical_publication =
            d_canonical_route_contributions != nullptr;
        dim3 scatter_grid(
            (N + kTileN - 1) / kTileN,
            k_partitions,
            num_active);
        dim3 route_reduce_grid(
            (N + kTileN - 1) / kTileN,
            num_active);
        dim3 direct_reduce_grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);

        // Narrow routes keep the same ordered partition arithmetic in their output tile.
        const bool fixed_short_pitch = direct_route_publication && N == kWideVerifierDownN &&
            K == kWideVerifierDownK && is_sm86_cuda_device(device_idx);
        const auto publish_down = [&]<uint8_t Codebook, bool FixedShortPitch>() {
            grouped_native_quantised_down_decode_publish_k32_kernel<Codebook, FixedShortPitch>
                <<<route_reduce_grid, block, 0, cuda_stream>>>(
                    d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids, d_weights,
                    d_canonical_route_contributions, N, K, num_experts);
        };
        const auto launch_down = [&]<uint8_t Codebook>() {
            if (direct_route_publication)
            {
                if constexpr (llaminar2::CUDAMoEBatchInvariantPolicy::qfnPackedShortDown(Codebook))
                {
                    if (fixed_short_pitch)
                        return publish_down.template operator()<Codebook, true>();
                }
                return publish_down.template operator()<Codebook, false>();
            }
            grouped_native_quantised_down_kpart_decode_route_kernel<Codebook>
                <<<scatter_grid, block, 0, cuda_stream>>>(
                    d_swiglu_int8, d_swiglu_scales, d_desc_table, d_expert_ids,
                    d_down_partials, num_active, N, K, num_experts, k_partitions);
        };
#define LAUNCH_GROUPED_DOWN_KPART(CB) launch_down.template operator()<CB>()

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN_KPART(0); break;
        case 4:  LAUNCH_GROUPED_DOWN_KPART(4); break;
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook: LAUNCH_GROUPED_DOWN_KPART(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook); break;
        case 5:  LAUNCH_GROUPED_DOWN_KPART(5); break;
        case 6:  LAUNCH_GROUPED_DOWN_KPART(6); break;
        case 7:  LAUNCH_GROUPED_DOWN_KPART(7); break;
        case 8:  LAUNCH_GROUPED_DOWN_KPART(8); break;
        case 9:  LAUNCH_GROUPED_DOWN_KPART(9); break;
        case 10: LAUNCH_GROUPED_DOWN_KPART(10); break;
        case 11: LAUNCH_GROUPED_DOWN_KPART(11); break;
        case 12: LAUNCH_GROUPED_DOWN_KPART(12); break;
        case 13: LAUNCH_GROUPED_DOWN_KPART(13); break;
        case 14: LAUNCH_GROUPED_DOWN_KPART(14); break;
        case 15: LAUNCH_GROUPED_DOWN_KPART(15); break;
        case 16: LAUNCH_GROUPED_DOWN_KPART(16); break;
        case 17: LAUNCH_GROUPED_DOWN_KPART(17); break;
        case 19: LAUNCH_GROUPED_DOWN_KPART(19); break;
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook: LAUNCH_GROUPED_DOWN_KPART(llaminar2::kNativeQuantisedExpandedInt8MinCodebook); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_DOWN_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN_KPART

        if (direct_route_publication)
            return finishLaunch("cudaMoE_grouped_swiglu_down_native_quantised_decode_publish_k32");
        if (!finishLaunch("cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart scatter"))
            return false;

        // Step 3: reduce — sum the k_partitions partials into the final output.
        if (canonical_publication)
        {
            grouped_native_quantised_down_kpart_route_reduce_kernel<<<
                route_reduce_grid, block, 0, cuda_stream>>>(
                d_down_partials,
                d_weights,
                d_canonical_route_contributions,
                num_active,
                N,
                k_partitions);
        }
        else
        {
            grouped_native_quantised_down_kpart_routes_reduce_kernel<<<
                direct_reduce_grid, block, 0, cuda_stream>>>(
                d_down_partials, d_weights, d_output,
                num_active, N, k_partitions);
        }

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_quantised_decode_table_kpart reduce");
    }

    bool cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart(
        const float *const *d_gate_ptrs,
        const float *const *d_up_ptrs,
        const void *d_runtime_layer,
        const int *d_expert_ids,
        const float *d_weights,
        int8_t *d_swiglu_int8,
        float *d_swiglu_scales,
        float *d_down_partials,
        float *d_output,
        int num_active,
        int d_model,
        int intermediate,
        int num_experts,
        uint8_t codebook_id,
        int k_partitions,
        int device_idx,
        void *stream)
    {
        const bool valid_k_partitions =
            k_partitions ==
            llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions;
        if (!d_gate_ptrs || !d_up_ptrs || !d_runtime_layer || !d_expert_ids || !d_weights ||
            !d_swiglu_int8 || !d_swiglu_scales || !d_down_partials || !d_output ||
            num_active <= 0 || d_model <= 0 || intermediate <= 0 || num_experts <= 0 ||
            (intermediate % 32) != 0 || !valid_k_partitions)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart] invalid arguments\n");
            return false;
        }

        cudaSetDevice(device_idx);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        grouped_swiglu_quantize_blockwise_kernel<<<num_active, kThreads, 0, cuda_stream>>>(
            d_gate_ptrs, d_up_ptrs, d_expert_ids, d_swiglu_int8, d_swiglu_scales, num_active, intermediate);
        if (!finishLaunch("cudaMoE_grouped_swiglu_runtime_quantize"))
            return false;

        constexpr int kTileN = 64;
        const int N = d_model;
        const int K = intermediate;
        dim3 scatter_grid(
            (N + kTileN - 1) / kTileN,
            k_partitions,
            num_active);
        dim3 reduce_grid((N + kTileN - 1) / kTileN);
        dim3 block(kTileN);
        const auto *runtime = static_cast<const DeviceMoELayerRuntimeView *>(d_runtime_layer);

#define LAUNCH_GROUPED_DOWN_RUNTIME_KPART(CB)                                                   \
    grouped_native_quantised_down_kpart_decode_runtime_kernel<CB><<<scatter_grid, block, 0, cuda_stream>>>( \
        d_swiglu_int8, d_swiglu_scales, runtime, d_expert_ids,                                  \
        d_down_partials, num_active, N, K, num_experts, k_partitions)

        switch (codebook_id)
        {
        case 0:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(0); break;
        case 4:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(4); break;
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook); break;
        case 5:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(5); break;
        case 6:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(6); break;
        case 7:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(7); break;
        case 8:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(8); break;
        case 9:  LAUNCH_GROUPED_DOWN_RUNTIME_KPART(9); break;
        case 10: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(10); break;
        case 11: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(11); break;
        case 12: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(12); break;
        case 13: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(13); break;
        case 14: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(14); break;
        case 15: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(15); break;
        case 16: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(16); break;
        case 17: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(17); break;
        case 19: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(19); break;
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(llaminar2::kNativeQuantisedExpandedInt8MinCodebook); break;
        case kMixedCodebookSentinel: LAUNCH_GROUPED_DOWN_RUNTIME_KPART(kMixedCodebookSentinel); break;
        default:
            std::fprintf(stderr, "[cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart] unsupported codebook_id=%u\n",
                         static_cast<unsigned>(codebook_id));
            return false;
        }

#undef LAUNCH_GROUPED_DOWN_RUNTIME_KPART

        if (!finishLaunch("cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart scatter"))
            return false;

        grouped_native_quantised_down_kpart_routes_reduce_kernel<<<
            reduce_grid, block, 0, cuda_stream>>>(
            d_down_partials, d_weights, d_output,
            num_active, N, k_partitions);

        return finishLaunch("cudaMoE_grouped_swiglu_down_native_quantised_decode_runtime_kpart reduce");
    }


    bool cudaMoE_reduce_canonical_route_contributions(
        const float *d_route_contributions,
        float *d_output,
        int seq_len,
        int top_k,
        int d_model,
        int device_idx,
        void *stream)
    {
        if (!d_route_contributions || !d_output || !stream ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kThreads = 256;
        const dim3 grid = make_row_major_grid(
            static_cast<unsigned int>((d_model + kThreads - 1) / kThreads),
            seq_len);
        reduce_canonical_route_contributions_kernel<<<
            grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            d_route_contributions, d_output, seq_len, top_k, d_model);
        return finishLaunch("cudaMoE_reduce_canonical_route_contributions");
    }

    bool cudaMoE_publish_shared_expert_rank_bank(
        const float *d_shared_output,
        float *d_canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *d_effective_seq_len,
        int device_idx,
        void *stream)
    {
        if (!d_shared_output || !d_canonical_publication || !stream ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0 || participant_index < 0 ||
            participant_index >= participant_count)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int kThreadsPerBlock = 256;
        const int vector_columns = (d_model + 3) / 4;
        const dim3 grid = make_row_major_grid(
            static_cast<unsigned int>(std::max(
                1,
                (vector_columns + kThreadsPerBlock - 1) /
                    kThreadsPerBlock)),
            seq_len * participant_count);
        publish_shared_expert_rank_bank_kernel<<<
            grid,
            kThreadsPerBlock,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            d_shared_output,
            d_canonical_publication,
            seq_len,
            top_k,
            d_model,
            participant_index,
            participant_count,
            d_effective_seq_len);
        return finishLaunch("cudaMoE_publish_shared_expert_rank_bank");
    }

    bool cudaMoE_finalize_canonical_publication(
        const float *d_input,
        const float *d_gate_inp,
        const float *d_canonical_publication,
        float *d_routed_output,
        float *d_shared_output,
        float *d_combined_output,
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        const int *d_effective_seq_len,
        int device_idx,
        void *stream)
    {
        if (!d_input || !d_gate_inp || !d_canonical_publication ||
            !d_routed_output || !d_shared_output || !d_combined_output ||
            !stream || seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        const dim3 grid = make_row_major_grid(
            static_cast<unsigned int>(
                (d_model + kCanonicalMoEOutputTileColumns - 1) /
                kCanonicalMoEOutputTileColumns),
            seq_len);
        finalize_canonical_moe_publication_kernel<<<
            grid,
            kThreads,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            d_input,
            d_gate_inp,
            d_canonical_publication,
            d_routed_output,
            d_shared_output,
            d_combined_output,
            seq_len,
            top_k,
            d_model,
            participant_count,
            d_effective_seq_len);
        return finishLaunch("cudaMoE_finalize_canonical_publication");
    }
} // extern "C"

namespace
{
#include "CUDARetainedHCFFNBody.cu.inc"
#include "CUDARetainedGroupedFFNBody.cu.inc"
} // namespace

namespace llaminar2::cuda::moe
{
    /** @copydoc RetainedGroupedFFNProgram::prepare */
    RetainedGroupedFFNProgram RetainedGroupedFFNProgram::prepare(DeviceId device,
        RetainedGroupedFFNRange range, void* stream, uint64_t generation)
    {
#if CUDART_VERSION >= 12080
        const auto native = static_cast<cudaStream_t>(stream);
        if (!device.is_cuda() || !native || native == cudaStreamLegacy || native == cudaStreamPerThread ||
            !generation || !range.rows || !range.live_rows || range.capacity <= 0 ||
            (range.width != kQFNDecodeGateUpN && range.width != 2 * kQFNDecodeGateUpN) ||
            (range.role != RetainedGroupedFFNRole::Routed && range.role != RetainedGroupedFFNRole::Shared))
            throw std::invalid_argument("Retained grouped QFN requires its original CUDA geometry and workspace");
        const auto& hardware = cudaHardwareMetadata().at(device.ordinal);
        int current = -1;
        if (hardware.sm_major != 8 || hardware.sm_minor != 6 || cudaGetDevice(&current) != cudaSuccess ||
            current != static_cast<int>(device.ordinal) ||
            range.capacity > hardware.maximum_grid_y)
            throw std::invalid_argument("Retained grouped QFN exceeds its admitted SM86 launch capacity");
        cudaStreamCaptureStatus capture{};
        if (cudaStreamIsCapturing(native, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
            throw std::logic_error("Retained grouped QFN preparation must precede native capture");

        const auto select = [&]<int Width>() {
            const void* gate = nullptr;
            const void* down = nullptr;
            const auto bind = [&]<uint8_t Codebook>() {
                if (Codebook == range.gate_up_codebook)
                    gate = reinterpret_cast<const void*>(retainedQFNGroupedGateUp<Codebook, Width>);
                if (Codebook == range.down_codebook)
                    down = range.role == RetainedGroupedFFNRole::Shared
                        ? reinterpret_cast<const void*>(retainedQFNGroupedDown<Codebook, Width, RetainedGroupedFFNRole::Shared>)
                        : reinterpret_cast<const void*>(retainedQFNGroupedDown<Codebook, Width, RetainedGroupedFFNRole::Routed>);
            };
            // Sources that share one packed execution codebook instantiate the
            // same math. The source identity remains with the prepared matrix.
            [&]<std::size_t... Source>(std::index_sequence<Source...>) {
                (bind.template operator()<native_quantised_formats::kAllSourceFormats[Source].metadata->codebook_id>(), ...);
            }(std::make_index_sequence<native_quantised_formats::kAllSourceFormats.size()>{});
            bind.template operator()<kNativeQuantisedExpandedInt8MinCodebook>();
            bind.template operator()<kNativeQuantisedIQ4XSSourceScaleCodebook>();
            const std::array entries{std::pair{gate, kThreads}, std::pair{down, retained_grouped_down_threads}};
            for (const auto& [function, threads] : entries)
            {
                cudaFuncAttributes resources{};
                if (!function || cudaFuncGetAttributes(&resources, function) != cudaSuccess ||
                    resources.maxThreadsPerBlock < threads || resources.localSizeBytes)
                    throw std::invalid_argument("Retained grouped QFN lacks a spill-free entry for its prepared decoder");
            }
            return RetainedGroupedFFNProgram(range, gate, down, stream, generation);
        };
        return range.width == kQFNDecodeGateUpN ? select.template operator()<kQFNDecodeGateUpN>()
            : select.template operator()<2 * kQFNDecodeGateUpN>();
#else
        (void)device; (void)range; (void)stream; (void)generation;
        throw std::invalid_argument("Retained grouped QFN requires CUDA device-updatable graph nodes");
#endif
    }

    /** @copydoc RetainedGroupedFFNProgram::record */
    void RetainedGroupedFFNProgram::record(void* stream, uint64_t generation) const
    {
#if CUDART_VERSION >= 12080
        if (stream != stream_ || generation != generation_)
            throw std::logic_error("Retained grouped QFN received a foreign stream or stale workspace");
        const auto native = static_cast<cudaStream_t>(stream);
        auto* rows = range_.rows;
        void* arguments[]{&rows};
        const std::array launches{
            cudaKernelNodeParams{.func = const_cast<void*>(gate_), .gridDim = dim3(1), .blockDim = dim3(kThreads),
                .kernelParams = arguments},
            cudaKernelNodeParams{.func = const_cast<void*>(down_), .gridDim = dim3(1),
                .blockDim = dim3(retained_grouped_down_threads),
                .kernelParams = arguments},
        };
        appendCUDADeviceSizedKernels(native, launches, [&](auto handles) {
            sizeRetainedQFNGroupedFFN<<<1, 1, 0, native>>>(handles[0], handles[1], range_.live_rows,
                range_.capacity, range_.width / 32);
            return cudaPeekAtLastError() == cudaSuccess;
        });
#else
        (void)stream; (void)generation;
        throw std::logic_error("Retained grouped QFN requires CUDA device-updatable graph nodes");
#endif
    }

    /** @brief Resolve the compiled original arithmetic and requested native phase. */
    static const void* retainedQFNM1PhaseFunction(uint8_t gate, uint8_t down,
        FullNativeHCFFNOutputDemand demand, bool observe, RoutedExpertRowExecutionPolicy role,
        RetainedM1NativePhase phase)
    {
        namespace formats = llaminar2::native_quantised_formats;
        const bool activated = demand.contains(FullNativeHCFFNOutputRole::HCActivated);
        const bool up = demand.contains(FullNativeHCFFNOutputRole::HCUp);
        const auto entry = [&]<uint8_t Gate, uint8_t Down, bool Observe, bool Shard>() -> const void* {
            if (phase == RetainedM1NativePhase::FFNSuffix)
                return reinterpret_cast<const void*>(retainedQFNM1HCFFNKernel<Gate, Down,
                    false, false, false, Shard, RetainedM1NativePhase::FFNSuffix>);
            if (phase == RetainedM1NativePhase::RoutePrefix)
            {
                if (activated)
                    return up ? reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<true, true>)
                              : reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<true, false>);
                return up ? reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<false, true>)
                          : reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<false, false>);
            }
            if (activated)
                return up ? reinterpret_cast<const void*>(retainedQFNM1HCFFNKernel<Gate, Down, Observe, true, true, Shard>)
                          : reinterpret_cast<const void*>(retainedQFNM1HCFFNKernel<Gate, Down, Observe, true, false, Shard>);
            return up ? reinterpret_cast<const void*>(retainedQFNM1HCFFNKernel<Gate, Down, Observe, false, true, Shard>)
                      : reinterpret_cast<const void*>(retainedQFNM1HCFFNKernel<Gate, Down, Observe, false, false, Shard>);
        };
        const auto select = [&]<bool Observe, bool Shard>() -> const void* {
            if (gate == formats::Q4_K.codebook_id && down == formats::Q5_1.codebook_id)
                return entry.template operator()<formats::Q4_K.codebook_id, formats::Q5_1.codebook_id, Observe, Shard>();
            if (gate == formats::Q4_K.codebook_id && down == formats::Q8_0.codebook_id)
                return entry.template operator()<formats::Q4_K.codebook_id, formats::Q8_0.codebook_id, Observe, Shard>();
            if (gate == formats::Q5_K.codebook_id && down == formats::Q8_0.codebook_id)
                return entry.template operator()<formats::Q5_K.codebook_id, formats::Q8_0.codebook_id, Observe, Shard>();
            return nullptr;
        };
        if (role == RoutedExpertRowExecutionPolicy::IntermediateShardLocal)
            return observe ? select.template operator()<true, true>() : select.template operator()<false, true>();
        if (role == RoutedExpertRowExecutionPolicy::ParticipantAssigned || role == RoutedExpertRowExecutionPolicy::FullyReplicatedLocal)
            return observe ? select.template operator()<true, false>() : select.template operator()<false, false>();
        return nullptr;
    }

    /** @copydoc retainedQFNM1Function */
    const void* retainedQFNM1Function(uint8_t gate, uint8_t down,
        FullNativeHCFFNOutputDemand demand, bool observe, RoutedExpertRowExecutionPolicy role)
    {
        return retainedQFNM1PhaseFunction(gate, down, demand, observe, role, RetainedM1NativePhase::Complete);
    }

    /** @copydoc RetainedM1Program::prepare */
    RetainedM1Program RetainedM1Program::prepare(DeviceId device, const RetainedM1HCFFN& arguments,
        uint8_t gate, uint8_t down, FullNativeHCFFNOutputDemand demand, bool observe, uint64_t generation,
        RetainedM1PacketOrdering ordering)
    {
        if (!device.is_cuda())
            throw std::invalid_argument("Retained QFN HC/FFN requires its original CUDA participant");
        const auto& hardware = cudaHardwareMetadata().at(device.ordinal);
        if (hardware.sm_major != 8 || hardware.sm_minor != 6)
            throw std::invalid_argument("Retained QFN HC/FFN has measured residency only on SM86");

        constexpr int hidden = kQFNDecodeGateUpK;
        constexpr int hc_width = kQFNDecodeGateUpN;
        const int shared_width = arguments.shared.gate.n;
        const bool ordered_readers = ordering == RetainedM1PacketOrdering::NativeCompletion;
        const auto& hc = arguments.hc;
        const auto& router = arguments.router;
        const auto valid_native_quant = [](const DeviceNativeQuantisedMatrixDesc& descriptor, int n, int k) {
            return descriptor.valid() && descriptor.n == n && descriptor.k == k &&
                descriptor.blocks_per_row == unsigned(k / 32) &&
                descriptor.source_identity_present == 1 &&
                (retainedQFNM1SuffixGateUpShard0(descriptor.codebook_id) != nullptr ||
                 retainedQFNM1SuffixGateUpShard1(descriptor.codebook_id) != nullptr);
        };
        const auto is_q8_matrix = [](const DeviceNativeQuantisedMatrixDesc& descriptor) {
            return descriptor.codebook_id == native_formats::Q8_0.codebook_id &&
                descriptor.source_identity_present == 1 &&
                descriptor.source_codebook_id == native_formats::Q8_0.codebook_id &&
                !descriptor.source_is_superblock;
        };
        const auto valid_workspace = [](const RetainedM1FFNWorkspace& work) {
            return work.down_codes && work.down_scales;
        };
        const auto same_row = [&](const DeviceRowRange& range) {
            return range.capacity() == 1 && range.countOwner() == hc.norm.extent.liveRows;
        };
        const bool routed_width_ok = ordered_readers
            ? (arguments.routed_width > 0 && (arguments.routed_width % 32) == 0 && arguments.routed_width <= 4096)
            : (arguments.routed_width == hc_width);
        const bool shared_width_ok = ordered_readers
            ? (shared_width > 0 && (shared_width % 32) == 0 && shared_width <= 4096)
            : (shared_width == hc_width);
        const bool hc_all_q8 = is_q8_matrix(hc.down) && is_q8_matrix(hc.up.weights);
        const bool shared_all_q8 = is_q8_matrix(arguments.shared.gate) &&
            is_q8_matrix(arguments.shared.up) && is_q8_matrix(arguments.shared.down);
        if (hc.write.extent.rows != 1 || hc.norm.extent.rows != 1 ||
            hc.write.extent.liveRows != hc.norm.extent.liveRows || !same_row(hc.up.rows) ||
            !same_row(hc.down_publication.rows) || hc.norm.hidden != hidden || hc.write.hidden != hidden ||
            !routed_width_ok || !shared_width_ok || hc.down_publication.columns != hc_width ||
            router.experts != kRetainedQFNExpertCount || router.top_k <= 0 || router.top_k > kMaxTopK ||
            !valid_native_quant(hc.down, hc_width, 4 * hidden) || !valid_native_quant(hc.up.weights, 4 * hidden, hc_width) ||
            !valid_native_quant(arguments.shared.gate, shared_width, hidden) || !valid_native_quant(arguments.shared.up, shared_width, hidden) ||
            !valid_native_quant(arguments.shared.down, hidden, shared_width) ||
            (!ordered_readers && (!hc_all_q8 || !shared_all_q8)) ||
            (static_cast<bool>(hc.injection_weights) != static_cast<bool>(hc.injection_output)) ||
            !std::isfinite(hc.norm.epsilon) || hc.norm.epsilon <= 0.0f)
            throw std::invalid_argument("Retained QFN HC/FFN has a mismatched original row or projection geometry");

        if (!hc.write.residual || !hc.write.block || !hc.write.gateLogits || !hc.write.output ||
            hc.norm.input != hc.write.output || !hc.norm.gain || !hc.norm.output ||
            !hc.norm_codes || !hc.norm_scales || !hc.raw_down || hc.up.input != hc.raw_down || !hc.mixed ||
            !hc.up.primary || !hc.up.residual || !hc.up.primaryScales || !hc.up.residualScales ||
            hc.up.primary != hc.down_publication.primary || hc.up.residual != hc.down_publication.residual ||
            hc.up.primaryScales != hc.down_publication.primaryScales ||
            hc.up.residualScales != hc.down_publication.residualScales ||
            hc.up.quantizer != Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal ||
            (demand.contains(FullNativeHCFFNOutputRole::HCActivated) && !hc.down_publication.activated) ||
            (demand.contains(FullNativeHCFFNOutputRole::HCUp) && !hc.up.output) ||
            !router.runtime || !router.weights || !router.logits || !router.route_indices || !router.route_weights ||
            !arguments.selected || !arguments.route_order || !arguments.hidden_codes || !arguments.hidden_scales ||
            (router.local_policy == RoutedExpertRowExecutionPolicy::IntermediateShardLocal && !arguments.route_plan_ready) ||
            !arguments.routed_contributions || !arguments.shared_contribution ||
            !valid_workspace(arguments.routed_workspace) || !valid_workspace(arguments.shared_workspace))
            throw std::invalid_argument("Retained QFN HC/FFN is missing an original graph buffer or prepared source");

        if ((arguments.route_publication != RetainedM1RoutePublication::Raw &&
             arguments.route_publication != RetainedM1RoutePublication::Preweighted) ||
            (arguments.shared_publication != RetainedM1SharedPublication::PartialBeforeCollective &&
             (arguments.shared_publication != RetainedM1SharedPublication::GatedComplete ||
              !arguments.shared_gate_input || !arguments.shared_gate_factor)))
            throw std::invalid_argument("Retained QFN HC/FFN contribution publication conflicts with its graph");

        constexpr int hc_packets = (4 * hidden) / 32;
        const auto partitions = hc.down_partitions;
        if (partitions.count <= 0 || partitions.count > hc_packets ||
            partitions.blocks_per_partition != (hc_packets + partitions.count - 1) / partitions.count ||
            (hc.down_fold == HCDownFold::Direct ? partitions.count != 1 :
                hc.down_fold != HCDownFold::OrderedPartitions || !hc.down_partials))
            throw std::invalid_argument("Retained QFN HC/FFN requires the original CUDA HC Down reduction geometry");

        if (ordered_readers && hc.norm.extent.liveRows)
            throw std::invalid_argument("CUDA native route completion requires the original fully-active row");
        if (ordered_readers && (!arguments.suffix_partials || !arguments.suffix_done))
            throw std::invalid_argument("CUDA native route completion requires suffix split-K workspace");

        RetainedM1HCFFN sealed_arguments = arguments;
        sealed_arguments.gate_codebook = gate;
        sealed_arguments.down_codebook = down;
        if (sealed_arguments.gate_promoted_codebook == 0xffu &&
            (gate == native_formats::Q5_K.codebook_id || gate == native_formats::Q4_K.codebook_id))
        {
            sealed_arguments.gate_promoted_codebook = kNativeQuantisedExpandedInt8MinCodebook;
            sealed_arguments.gate_source_superblock = 1u;
        }
        if (sealed_arguments.down_promoted_codebook == 0xffu &&
            (down == native_formats::Q5_1.codebook_id || down == native_formats::Q4_1.codebook_id))
        {
            sealed_arguments.down_promoted_codebook = kNativeQuantisedExpandedInt8MinCodebook;
            sealed_arguments.down_source_superblock = 0u;
        }
        // Only one of gate/down promotion may split the single canonical_count partition.
        if (sealed_arguments.gate_promoted_codebook != 0xffu &&
            sealed_arguments.down_promoted_codebook != 0xffu)
        {
            // Keep down promotion when down is Q5_1 (Q4_K/Q5_1 layers), else gate promotion.
            if (down == native_formats::Q5_1.codebook_id)
                sealed_arguments.gate_promoted_codebook = 0xffu;
            else
                sealed_arguments.down_promoted_codebook = 0xffu;
        }

        // Declaratively reserve up to 8 SMs so co-resident graph service kernels
        // (MappedTransferService, MoEOverlayDeviceControllerGraphService, outbound
        // mapped dispatch batch, and NCCL collectives) never starve or deadlock
        // cooperative grid.sync() on continuation root devices (e.g. CUDA:0).
        const auto co_residency = NativeCoResidencyReservation::reserveProcessors(
            hardware.sm_count > 16 ? 8u : (hardware.sm_count > 4 ? 4u : (hardware.sm_count > 2 ? 2u : 0u)));
        const auto select_phase = [&](RetainedM1NativePhase phase) {
            const void* entry = nullptr;
            if (phase == RetainedM1NativePhase::RoutePrefix)
            {
                const bool activated = demand.contains(FullNativeHCFFNOutputRole::HCActivated);
                const bool up = demand.contains(FullNativeHCFFNOutputRole::HCUp);
                if (hc_all_q8)
                {
                    if (activated)
                        entry = up ? reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<true, true, false>)
                                   : reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<true, false, false>);
                    else
                        entry = up ? reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<false, true, false>)
                                   : reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<false, false, false>);
                }
                else
                {
                    if (activated)
                        entry = up ? reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<true, true, true>)
                                   : reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<true, false, true>);
                    else
                        entry = up ? reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<false, true, true>)
                                   : reinterpret_cast<const void*>(retainedQFNM1HCRouterKernel<false, false, true>);
                }
            }
            else
            {
                entry = retainedQFNM1PhaseFunction(gate, down, demand, observe, router.local_policy, phase);
            }
            return NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::prepare(
                device, entry, kThreads, 0, generation,
                NativeResidentLaunchPolicy::Cooperative, co_residency);
        };
        const auto admitted = select_phase(ordered_readers ? RetainedM1NativePhase::RoutePrefix
                                                          : RetainedM1NativePhase::Complete);
        const auto route_prefix = ordered_readers ? std::optional{admitted} : std::nullopt;
        const void* route_plan = nullptr;
        const void* suffix_gate_up = nullptr;
        const void* suffix_down = nullptr;
#if CUDART_VERSION >= 12080
        if (ordered_readers)
        {
            const bool is_shard = router.local_policy == RoutedExpertRowExecutionPolicy::IntermediateShardLocal;
            if (is_shard)
                route_plan = observe ? reinterpret_cast<const void*>(retainedQFNM1TopKPlanKernel<true, true>)
                                     : reinterpret_cast<const void*>(retainedQFNM1TopKPlanKernel<false, true>);
            else
                route_plan = observe ? reinterpret_cast<const void*>(retainedQFNM1TopKPlanKernel<true, false>)
                                     : reinterpret_cast<const void*>(retainedQFNM1TopKPlanKernel<false, false>);
            const bool shared_gu_q8 =
                arguments.shared.gate.codebook_id == native_formats::Q8_0.codebook_id &&
                arguments.shared.up.codebook_id == native_formats::Q8_0.codebook_id;
            const bool shared_down_q8 =
                arguments.shared.down.codebook_id == native_formats::Q8_0.codebook_id;
            const uint8_t suffix_gate_key = shared_gu_q8 ? gate : uint8_t{0xFFu};
            const uint8_t suffix_down_key = shared_down_q8 ? down : uint8_t{0xFFu};
            suffix_gate_up = retainedQFNM1SuffixGateUpShard0(suffix_gate_key);
            if (!suffix_gate_up)
                suffix_gate_up = retainedQFNM1SuffixGateUpShard1(suffix_gate_key);
            suffix_down = retainedQFNM1SuffixDownShard0(suffix_down_key);
            if (!suffix_down)
                suffix_down = retainedQFNM1SuffixDownShard1(suffix_down_key);
            const std::array suffix_entries{
                std::pair{route_plan, static_cast<unsigned>(kThreads)},
                std::pair{suffix_gate_up, retained_suffix_gate_up_threads},
                std::pair{suffix_down, retained_suffix_down_threads},
            };
            for (const auto& [fn, threads] : suffix_entries)
            {
                cudaFuncAttributes attr{};
                if (!fn || cudaFuncGetAttributes(&attr, fn) != cudaSuccess ||
                    attr.maxThreadsPerBlock < static_cast<int>(threads) || attr.localSizeBytes)
                    throw std::invalid_argument("CUDA retained QFN suffix requires spill-free device-sized entries");
            }
        }
#endif
        const unsigned available_sms = admitted.availableProcessors();
        auto blocks = std::min(admitted.maximumGrid(),
            std::min(2u * available_sms, hidden / retained_hc_read_warp_channels));
        const unsigned prefix_blocks = ordered_readers
            ? route_prefix->maximumGrid()
            : 0u;
        if (route_prefix && route_prefix->resources().private_bytes)
            throw std::invalid_argument("CUDA retained QFN route prefix requires spill-free native arithmetic");
        if (!ordered_readers && blocks > admitted.maximumGrid())
            throw std::invalid_argument("Retained QFN HC/FFN entry cannot keep the measured CTA grid resident");
        if (route_prefix && prefix_blocks < 2u)
            throw std::invalid_argument("Retained QFN HC/FFN route prefix cannot keep its CTA grid resident");
        if (router.local_policy == RoutedExpertRowExecutionPolicy::IntermediateShardLocal && blocks < 2u)
            throw std::invalid_argument("Retained QFN HC/FFN needs separate route and shared Gate/Up CTAs");
        return {admitted, sealed_arguments, blocks, route_prefix, prefix_blocks, route_plan, suffix_gate_up, suffix_down};
    }

    /** @copydoc RetainedM1Program::launch */
    void RetainedM1Program::launch(void* stream, uint64_t generation, const std::function<void()>& record_ready) const
    {
        const auto native = static_cast<cudaStream_t>(stream);
        if (!native || native == cudaStreamLegacy || native == cudaStreamPerThread || generation != admission_.generation())
            throw std::invalid_argument("Retained QFN HC/FFN received a default stream or a stale workspace generation");
        auto arguments = arguments_;
        void* parameters[]{&arguments};
        const auto* count = arguments.hc.norm.extent.liveRows;
        if (!route_prefix_.has_value() && bool(record_ready))
            throw std::logic_error("Retained QFN complete program cannot export an early native route frontier");
        if (route_prefix_)
        {
#if CUDART_VERSION >= 12080
            cudaStreamCaptureStatus status{};
            if (cudaStreamIsCapturing(native, &status) != cudaSuccess || status != cudaStreamCaptureStatusActive)
                throw std::logic_error("Retained QFN native route completion requires its active graph");
            // Unified resident suffix for both OrderedTimelineRecording (2xCUDA+2xCPU) and
            // cloneable full-graph capture (2xCUDA+4xROCm): every CTA is sized to the exact
            // guaranteed live shared-expert work (shared_packets * 8 = 80 CTAs for Gate/Up,
            // hidden / 32 = 80 CTAs for Down) and grid-strides compact local experts without
            // device-updatable graph node handles (eliminating ~2 ms/tok cudaGraphLaunch
            // upload/re-sync overhead and ~8 us/layer device-side grid-update latency).
            arguments.suffix_gate_up_node = nullptr;
            arguments.suffix_down_node = nullptr;
            arguments.suffix_resident_loop = 1u;
            NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::launch(
                *route_prefix_, parameters, prefix_blocks_, native, generation);
            if (cudaLaunchKernel(route_plan_, dim3(1), dim3(kThreads), parameters, 0, native) != cudaSuccess)
                throw std::runtime_error("CUDA retained QFN TopK plan launch failed");
            if (record_ready)
                record_ready();
            const unsigned shared_gate_up_tasks =
                retained_suffix_gate_up_k_splits * static_cast<unsigned>(arguments.shared.gate.n / 32);
            // Size SuffixGateUp to exact guaranteed live shared-expert tasks (capped at 2 CTAs/SM
            // of availableProcessors) and SuffixDown to hidden / 16 (160 CTAs x 10 half-warps =
            // 10 warps/SM) while leaving >1,000 free threads/SM for instant co-resident dispatch.
            const unsigned gate_up_grid =
                std::min(shared_gate_up_tasks, route_prefix_->availableProcessors() * 2u);
            if (cudaLaunchKernel(suffix_gate_up_, dim3(gate_up_grid, 1),
                    dim3(retained_suffix_gate_up_threads), parameters, 0, native) != cudaSuccess ||
                cudaLaunchKernel(suffix_down_, dim3(kQFNDecodeGateUpK / 16, 1),
                    dim3(retained_suffix_down_threads), parameters, 0, native) != cudaSuccess)
                throw std::runtime_error("CUDA retained QFN resident suffix launch failed");
            return;
#else
            throw std::logic_error("CUDA retained QFN native route completion requires CUDA 12.8+");
#endif
        }
        if (!count)
        {
            NativeCooperativeKernelBoundary<NativeCooperativeBackend::CUDA>::launch(
                admission_, parameters, blocks_, native, generation);
            return;
        }

        CUDAActiveCaptureConditional branch(native, CUDAActiveCaptureConditionalKind::IfOnly);
        const cudaKernelNodeParams launch{.func = const_cast<void*>(admission_.function()),
            .gridDim = dim3(blocks_), .blockDim = dim3(admission_.threads()), .sharedMemBytes = 0,
            .kernelParams = parameters, .extra = nullptr};
        if (!branch.publishPredicate([&](cudaGraphConditionalHandle condition) {
                retainedQFNM1Predicate<<<1, 32, 0, native>>>(condition, count, arguments.router);
                return cudaPeekAtLastError() == cudaSuccess;
            }) || !branch.beginBranches() ||
            !branch.appendKernel(CUDAActiveCaptureConditionalBranch::IfNonZero, launch,
                "QFN retained HC/FFN", CUDACapturedKernelGrid::Cooperative) || !branch.commit())
            throw std::runtime_error(branch.error());
    }
}
