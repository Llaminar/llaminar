/**
 * @file CUDAMoEGroupedPrefillLaunchKernels.cu
 * @brief CUDA MoE grouped NativeQuantised prefill kernels, resource inspection, and prefill pipeline launcher.
 */
#include "CUDAMoENativeQuantisedDecodeHelpers.cuh"

namespace
{
    // Independent warps preserve the canonical 32-value quantization tree.
    static constexpr int kWarpsPerQuantBlock = 8; // 8 warps = 256-thread block
    /**
     * @brief Quantize each original hidden row once when the router has not.
     * @param hidden Original FP32 token rows, without expert-route duplication.
     * @param A_int8 Existing arena storage for canonical original Q8 rows.
     * @param scales_A_blockwise Existing arena FP32 scales, one per 32 values.
     * @param source_rows Physical token-row count in this captured invocation.
     * @param K Hidden width, a positive multiple of the 32-value contract.
     *
     * The paired gate/up kernels resolve grouped routes into these rows. A live
     * router publication skips this producer entirely, with its exact stream
     * edge already owned by the enclosing graph.
     */
    __global__ void moe_prefill_hidden_quantize_rows_kernel(
        const float *__restrict__ hidden,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int source_rows,
        int K)
    {
        const int slot = logical_row_from_grid_yz();
        if (slot >= source_rows)
            return;

        // Map this thread to (warp, lane); each warp handles a distinct 32-col block.
        const int warp = threadIdx.x >> 5;  // 0..kWarpsPerQuantBlock-1
        const int lane = threadIdx.x & 31;  // 0..31
        const int block_idx = blockIdx.x * kWarpsPerQuantBlock + warp;
        const int col = block_idx * 32 + lane;
        const int blocks_per_row = (K + 31) / 32;
        if (block_idx >= blocks_per_row || col >= K)
            return;

        const float value = hidden[static_cast<size_t>(slot) * K + col];

        // Warp-local absmax reduction (each warp owns its own 32-element block).
        float abs_value =
            llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                value, 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value =
                llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                    abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale =
            llaminar2::device_q8_activation_contract::scale(abs_value);
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx] = scale;

        A_int8[static_cast<size_t>(slot) * K + col] =
            static_cast<int8_t>(
                llaminar2::device_q8_activation_contract::quantize(
                    value, scale));
    }


    // Fused gate/up GEMM + SwiGLU + blockwise int8 quantization.
    //
    // This is the fused-epilogue variant of grouped_native_quantised_gate_up_prefill_kernel: after
    // computing gate_acc/up_acc for each token in the tile, it directly evaluates
    // silu(gate)*up and blockwise-quantizes the result to int8, writing the down-projection
    // input (swiglu_int8 + swiglu_scales) in place. This eliminates the FP32 gate/up global
    // round-trip and the separate grouped_prefill_swiglu_quantize_blockwise_kernel launch.
    //
    // Quantization layout: each warp (32 lanes) spans exactly one aligned 32-wide block of the
    // intermediate (N) dimension (kTileN=128 is a multiple of 32, and N % 32 == 0), so the
    // per-block absmax is a warp-shuffle reduction with no cross-warp synchronization.
    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_quantised_gate_up_swiglu_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        llaminar2::MoEGroupedSourceRows source_rows,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int N,
        int K)
    {
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Threads with n >= N still participate in cooperative staging + __syncthreads.
        const bool active = (n < N);

        const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
        if (!native_quantised_pair_shape_ok<CodebookId>(gate_desc, up_desc, N, K))
            return;

        const uint8_t *gate_payload_base = gate_desc.payload;
        const uint16_t *gate_scale_base = static_cast<const uint16_t *>(gate_desc.scales);
        const uint16_t *gate_min_base = static_cast<const uint16_t *>(gate_desc.mins);
        const uint32_t *gate_emin_base = static_cast<const uint32_t *>(gate_desc.emins);
        const uint8_t *up_payload_base = up_desc.payload;
        const uint16_t *up_scale_base = static_cast<const uint16_t *>(up_desc.scales);
        const uint16_t *up_min_base = static_cast<const uint16_t *>(up_desc.mins);
        const uint32_t *up_emin_base = static_cast<const uint32_t *>(up_desc.emins);

        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        static_assert(kTileM * 8 <= kTileN);
        const int a_source = threadIdx.x < tokens_in_group * 8
            ? source_rows.sourceRow(first_slot + (threadIdx.x >> 3)) : 0;
        const int scale_source = threadIdx.x < tokens_in_group
            ? source_rows.sourceRow(first_slot + threadIdx.x) : 0;

        float gate_acc[kTileM] = {};
        float up_acc[kTileM] = {};

        constexpr int kPayloadBytes =
            llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::payload_bytes;

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            // Hoisted weight decode (overlaps activation staging + barrier latency).
            int32_t gate_groups[8];
            int32_t up_groups[8];
            const uint8_t *gate_payload = gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload = up_payload_base + linear * kPayloadBytes;
            if (active)
            {
                llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(gate_payload, gate_groups);
                llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(up_payload, up_groups);
            }

            // Cooperatively stage the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int g = idx & 7;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(a_source) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(scale_source) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    gate_groups, gate_scale_base, gate_min_base, gate_emin_base, gate_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, gate_acc, gate_desc.codebook_id);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    up_groups, up_scale_base, up_min_base, up_emin_base, up_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, up_acc, up_desc.codebook_id);
            }

            __syncthreads();
        }

        // Fused SwiGLU + blockwise int8 quant epilogue. N == intermediate here.
        const int lane = threadIdx.x & 31;
        const int blocks_per_row_out = N / 32;
        const int quant_block = n >> 5; // aligned 32-wide block index along intermediate
#pragma unroll
        for (int m = 0; m < kTileM; ++m)
        {
            if (m >= tokens_in_group)
                break; // tokens_in_group is block-uniform → all lanes break together (shfl-safe).

            const int slot = first_slot + m;
            // Inactive lanes (n >= N) contribute 0 to the warp absmax and skip the write.
            const float value = active ? (silu(gate_acc[m]) * up_acc[m]) : 0.0f;

            float abs_value =

                llaminar2::device_q8_activation_contract::maximumAbsoluteValue(

                    value, 0.0f);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                abs_value =
                    llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                        abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

            const float scale =
                llaminar2::device_q8_activation_contract::scale(abs_value);
            if (active)
            {
                if (lane == 0)
                    swiglu_scales[static_cast<size_t>(slot) * blocks_per_row_out + quant_block] = scale;
                swiglu_int8[static_cast<size_t>(slot) * N + n] =
                    static_cast<int8_t>(
                        llaminar2::device_q8_activation_contract::quantize(
                            value, scale));
            }
        }
    }

    __global__ void grouped_prefill_swiglu_quantize_blockwise_kernel(
        const float *__restrict__ gate,
        const float *__restrict__ up,
        int8_t *__restrict__ A_int8,
        float *__restrict__ scales_A_blockwise,
        int total_slots,
        int K)
    {
        const int block_idx = blockIdx.x;
        const int slot = logical_row_from_grid_yz();
        const int lane = threadIdx.x;
        const int col = block_idx * 32 + lane;
        if (slot >= total_slots || lane >= 32 || col >= K)
            return;

        const size_t idx = static_cast<size_t>(slot) * K + col;
        const float g = gate[idx];
        const float value = silu(g) * up[idx];

        float abs_value =

            llaminar2::device_q8_activation_contract::maximumAbsoluteValue(

                value, 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            abs_value =
                llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                    abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));

        const float scale =
            llaminar2::device_q8_activation_contract::scale(abs_value);
        const int blocks_per_row = (K + 31) / 32;
        if (lane == 0)
            scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx] = scale;

        A_int8[idx] = static_cast<int8_t>(
            llaminar2::device_q8_activation_contract::quantize(
                value, scale));
    }

    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_quantised_down_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ output,
        int N,
        int K)
    {
        // kTileM tokens processed per block; larger values amortize weight decode (see gate_up).
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Per-thread output column may exceed N; such threads must still participate in the
        // cooperative staging + __syncthreads. Guard only weight decode / output writes.
        const bool active = (n < N);

        const DeviceNativeQuantisedMatrixDesc desc = descs[expert_id];
        if (!native_quantised_desc_shape_ok<CodebookId>(desc, N, K))
            return;

        const uint8_t *payload_base = desc.payload;
        const uint16_t *scale_base = static_cast<const uint16_t *>(desc.scales);
        const uint16_t *min_base = static_cast<const uint16_t *>(desc.mins);
        const uint32_t *emin_base = static_cast<const uint32_t *>(desc.emins);

        // Staged activation tile (see gate_up kernel) to remove redundant L1 loads.
        // 16-byte aligned for vectorized 2× int4 (128-bit) per-token reads.
        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        float acc[kTileM] = {};

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            // Cooperatively stage the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int m = idx >> 3;
                const int g = idx & 7;
                const int slot = first_slot + m;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(slot) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                const int slot = first_slot + m;
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(slot) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);
                int32_t packed_groups[8];
                const uint8_t *payload = payload_base +
                    linear * llaminar2::cuda_native_quantised::payload_bytes_for_codebook<CodebookId>();
                llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(payload, packed_groups);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    packed_groups, scale_base, min_base, emin_base, payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, acc, desc.codebook_id);
            }

            __syncthreads();
        }

        if (active)
        {
#pragma unroll
            for (int m = 0; m < kTileM; ++m)
            {
                if (m >= tokens_in_group)
                    break;
                const int slot = first_slot + m;
                output[static_cast<size_t>(slot) * N + n] = acc[m];
            }
        }
    }

    __global__ void grouped_prefill_scatter_weighted_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ grouped_token_indices,
        const float *__restrict__ grouped_weights,
        int total_slots,
        int max_tokens,
        int top_k,
        int grouped_indices_are_route_slots,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int slot = logical_row_from_grid_yz();
        if (slot >= total_slots || col >= d_model)
            return;

        const int source_index = grouped_token_indices[slot];
        const int token =
            grouped_indices_are_route_slots ? (source_index / top_k) : source_index;
        if (token < 0 || token >= max_tokens)
        {
            if (col == 0)
            {
                printf("grouped_prefill_invalid_scatter_token slot=%d index=%d token=%d "
                       "max_tokens=%d top_k=%d route_slot_mode=%d\\n",
                       slot, source_index, token, max_tokens, top_k,
                       grouped_indices_are_route_slots);
            }
            FAIL_FAST_INCOMPLETE_LLEP_TRANSFER(
                "grouped prefill scatter references an invalid destination token");
            return;
        }
        const float weight = grouped_weights[slot];
        const float value = expert_output[static_cast<size_t>(slot) * d_model + col];
        atomicAdd(output + static_cast<size_t>(token) * d_model + col, weight * value);
    }

    __global__ void grouped_prefill_scatter_weighted_ordered_kernel(
        float *__restrict__ output,
        const float *__restrict__ expert_output,
        const int *__restrict__ original_to_grouped,
        const float *__restrict__ grouped_weights,
        int seq_len,
        int top_k,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int token = logical_row_from_grid_yz();
        if (token >= seq_len || col >= d_model)
            return;

        float sum = 0.0f;
#pragma unroll 1
        for (int k = 0; k < top_k; ++k)
        {
            const int original_slot = token * top_k + k;
            const int grouped_slot = original_to_grouped[original_slot];
            if (grouped_slot < 0)
                continue;
            const float weight = grouped_weights[grouped_slot];
            const float value = expert_output[static_cast<size_t>(grouped_slot) * d_model + col];
            sum = moe_accumulate_rn(
                sum,
                moe_weight_route_rn(weight, value));
        }
        output[static_cast<size_t>(token) * d_model + col] = sum;
    }

    /**
     * @brief Fold already-weighted grouped rows in original router order.
     *
     * The grouped IMMA down projection applies a route weight independently to
     * every public-M1 K partition before reducing those partitions. Its output
     * is therefore the complete serial route contribution and must not be
     * multiplied a second time during token publication.
     */
    __global__ void grouped_prefill_scatter_contributions_ordered_kernel(
        float *__restrict__ output,
        const float *__restrict__ grouped_contributions,
        const int *__restrict__ original_to_grouped,
        int seq_len,
        int top_k,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int token = logical_row_from_grid_yz();
        if (token >= seq_len || col >= d_model)
            return;

        float sum = 0.0f;
#pragma unroll 1
        for (int route = 0; route < top_k; ++route)
        {
            const int grouped_slot =
                original_to_grouped[token * top_k + route];
            if (grouped_slot >= 0)
            {
                sum = moe_accumulate_rn(
                    sum,
                    grouped_contributions[
                        static_cast<size_t>(grouped_slot) * d_model + col]);
            }
        }
        output[static_cast<size_t>(token) * d_model + col] = sum;
    }

    /**
     * @brief Publish already-weighted grouped rows into canonical route slots.
     *
     * Missing/non-local routes overwrite their complete slot with zero. The
     * later collective and canonical reducer therefore observe the same route
     * dimension and FP32 order as serial decode on every participant.
     */
    __global__ void grouped_prefill_scatter_contributions_canonical_kernel(
        float *__restrict__ route_output,
        const float *__restrict__ grouped_contributions,
        const int *__restrict__ original_to_grouped,
        int total_slots,
        int d_model)
    {
        constexpr int kTileN = 64;
        const int col = blockIdx.x * kTileN + threadIdx.x;
        const int original_slot = logical_row_from_grid_yz();
        if (original_slot >= total_slots || col >= d_model)
            return;

        const int grouped_slot = original_to_grouped[original_slot];
        route_output[
            static_cast<size_t>(original_slot) * d_model + col] =
            grouped_slot >= 0
                ? grouped_contributions[
                      static_cast<size_t>(grouped_slot) * d_model + col]
                : 0.0f;
    }

    /**
     * @brief Compute a resolved physical dot product over [b_start, b_end).
     *
     * This is the split-K building block: each caller reduces only a slice of the
     * K dimension so that multiple thread blocks can cooperate on a single output
     * column. Passing [0, K/32) reproduces the full reduction.
     *
     * @param desc                 NativeQuantised weight descriptor (column n of B).
     * @param n                    Output column index into the weight matrix.
     * @param A_int8               Quantized activation row (int8, K elements).
     * @param scales_A_blockwise   Per-K-block activation scales.
     * @param N                    Weight matrix column count (stride for payload).
     * @param b_start              First K-block (inclusive) this call reduces.
     * @param b_end                Last K-block (exclusive) this call reduces.
     * @param iq3_table            Canonical grid already copied into CTA shared memory.
     * @param iq1m_correction      Original signed delta terms prepared from the K32 activation.
     * @return Partial accumulated dot product over the requested K-block range.
     * @tparam CodebookId Physical prepared weight format.
     * @tparam Order Direct Down folds each original rounded K32 partition.
     *         A partition partial retains the original within-partition fold.
     * @tparam FixedBlocks Full row length in K32 packets when the caller has
     *         admitted the fixed traversal. Zero uses the supplied interval.
     * @tparam Unroll Number of independent packets to issue per iteration.
     * @tparam SharedIQ3 Consume the CTA's prepared canonical table.
     * @tparam SharedIQ1MCorrection Consume the CTA's exact IQ1_M activation delta terms.
     * @pre The caller has resolved the physical codebook, validated the
     *      descriptor and 16-byte-aligned activation row, and clipped the K32
     *      interval.
     */
    template <uint8_t CodebookId, int Pitch = 0, int Packets = 0,
              NativeMoEDotOrder PacketOrder = NativeMoEDotOrder::PartitionPartial,
              int ExactTile = 0, unsigned RoutesPerToken = 0, bool PlanValidated = false>
    __device__ __forceinline__ void grouped_native_quantised_gate_up_ordered_kpart_scatter(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        llaminar2::MoEGroupedSourceRows source_rows,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int original_slot_base,
        int tile_route_slots,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        if constexpr (ExactTile > 0)
        {
            static_assert(ExactTile == 64 && Pitch == kQFNDecodeGateUpN && Pitch % ExactTile == 0 &&
                          Packets == kQFNDecodeGateUpPackets);
            k_partitions = llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
            N = Pitch;
            K = kQFNDecodeGateUpK;
        }
        const int tile_columns = ExactTile > 0 ? ExactTile : blockDim.x;
        const int n = blockIdx.x * tile_columns + threadIdx.x;
        const int local_route = blockIdx.y;
        const int k_part = blockIdx.z;
        if constexpr (ExactTile == 0)
        {
            if (n >= N || local_route >= tile_route_slots || k_part >= k_partitions)
                return;
        }

        const int original_slot = original_slot_base + local_route;
        const int grouped_slot = original_to_grouped[original_slot];
        const int expert_id = original_expert_ids[original_slot];
        if (grouped_slot < 0 || expert_id < 0 || expert_id >= num_experts)
            return;

        const int blocks_per_row = K / 32;
        const int blocks_per_part = Packets > 0 ? Packets :
            (blocks_per_row + k_partitions - 1) / k_partitions;
        const int b_start = k_part * blocks_per_part;
        int b_end = b_start + blocks_per_part;
        if constexpr (ExactTile == 0)
            b_end = min(b_end, blocks_per_row);

        const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
        if constexpr (PlanValidated)
            static_assert(ExactTile > 0);
        else if (!native_quantised_pair_shape_ok<CodebookId>(gate_desc, up_desc, N, K))
            return;

        const size_t partial_index =
            (static_cast<size_t>(local_route) * static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);
        if constexpr (ExactTile == 0)
        {
            if (b_start >= b_end)
            {
                gate_partials[partial_index] = 0.0f;
                up_partials[partial_index] = 0.0f;
                return;
            }
        }

        const int source_row = [&] {
            if constexpr (RoutesPerToken > 0)
            {
                static_assert(ExactTile > 0);
                return static_cast<int>(static_cast<unsigned>(original_slot) / RoutesPerToken);
            }
            else
                return source_rows.sourceRow(grouped_slot);
        }();
        const int8_t *slot_A = A_int8 + static_cast<size_t>(source_row) * K;
        const float *slot_scales =
            scales_A_blockwise + static_cast<size_t>(source_row) * blocks_per_row;
        float gate_partial = 0.0f;
        float up_partial = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
        if constexpr (Pitch > 0 && llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::is_iq4_lut)
        {
            moe_iq4_interleaved_gate_up<Pitch, Packets>(
                gate_desc, up_desc, slot_A, slot_scales, n, b_start, gate_partial, up_partial);
        }
        else
#endif
        if constexpr (Pitch > 0)
        {
            constexpr bool split_iq1_s = ExactTile > 0 && CodebookId == native_formats::IQ1_S.codebook_id;
            constexpr auto signs = [] {
                if constexpr (ExactTile > 0 && CodebookId == native_formats::IQ2_XS.codebook_id)
                {
                    if constexpr (PlanValidated)
                        return NativeMoEIQ2SignSchedule::ClampedFunnelAndIndependentProducts;
                    return NativeMoEIQ2SignSchedule::IndependentProducts;
                }
                return NativeMoEIQ2SignSchedule::RegisterShifts;
            }();
            native_quantised_gate_up_fixed_partition<CodebookId, Pitch, Packets, PacketOrder, split_iq1_s, signs>(
                gate_desc, up_desc, slot_A, slot_scales, n, b_start, gate_partial, up_partial);
        }
        else
        {
            native_quantised_dot_desc_pair_range<CodebookId>(
                gate_desc, up_desc, n, slot_A, slot_scales, N, K,
                b_start, b_end, gate_partial, up_partial);
        }
        gate_partials[partial_index] = gate_partial;
        up_partials[partial_index] = up_partial;
    }

    /** @brief Codebooks measured with a 16-CTA budget and zero spills on SM86. */
    template <uint8_t CodebookId>
    inline constexpr bool kQFNBoundedGateRegisters =
        CodebookId == native_formats::IQ1_S.codebook_id ||
        CodebookId == native_formats::IQ1_M.codebook_id ||
        CodebookId == native_formats::IQ2_S.codebook_id ||
        CodebookId == native_formats::IQ2_XS.codebook_id;

    /**
     * @brief Launch the ordered scatter with the general register budget.
     * @copydetails grouped_native_quantised_gate_up_ordered_kpart_scatter
     */
    template <uint8_t CodebookId, int Pitch = 0, int Packets = 0,
              NativeMoEDotOrder PacketOrder = NativeMoEDotOrder::PartitionPartial,
              int ExactTile = 0, unsigned RoutesPerToken = 0, bool PlanValidated = false>
    __global__ void grouped_native_quantised_gate_up_ordered_kpart_scatter_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        llaminar2::MoEGroupedSourceRows source_rows,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int original_slot_base,
        int tile_route_slots,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        grouped_native_quantised_gate_up_ordered_kpart_scatter<
            CodebookId, Pitch, Packets, PacketOrder, ExactTile, RoutesPerToken, PlanValidated>(
            A_int8, scales_A_blockwise, source_rows, gate_descs, up_descs,
            original_to_grouped, original_expert_ids, gate_partials, up_partials,
            original_slot_base, tile_route_slots, N, K, num_experts, k_partitions);
    }

    /**
     * @brief Bound registers only for measured, plan-validated QFN tiles.
     * @pre Capture submits every original output column in 64-thread blocks.
     *      The upstream producer selected a uniform measured physical codebook.
     * @copydetails grouped_native_quantised_gate_up_ordered_kpart_scatter
     */
    template <uint8_t CodebookId, int Pitch, int Packets, NativeMoEDotOrder PacketOrder,
              int ExactTile, unsigned RoutesPerToken, bool PlanValidated>
    __global__ __launch_bounds__(64, 16)
    void grouped_native_quantised_gate_up_ordered_kpart_scatter_kernel_bounded(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        llaminar2::MoEGroupedSourceRows source_rows,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        float *__restrict__ gate_partials,
        float *__restrict__ up_partials,
        int original_slot_base,
        int tile_route_slots,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        static_assert(ExactTile == 64 && PlanValidated && kQFNBoundedGateRegisters<CodebookId>);
        grouped_native_quantised_gate_up_ordered_kpart_scatter<
            CodebookId, Pitch, Packets, PacketOrder, ExactTile, RoutesPerToken, PlanValidated>(
            A_int8, scales_A_blockwise, source_rows, gate_descs, up_descs,
            original_to_grouped, original_expert_ids, gate_partials, up_partials,
            original_slot_base, tile_route_slots, N, K, num_experts, k_partitions);
    }


    /**
     * @brief Evaluate the fixed QFN SiLU with one exponential and reciprocal chain.
     * @param gate Ordered FP32 Gate value, including zero and exceptional values.
     * @return The canonical rounded SiLU word before multiplication by Up.
     */
    __device__ __forceinline__ float qfn_shared_sigmoid_silu(float gate)
    {
        using namespace llaminar2::device_swiglu_contract;
        const bool nonnegative = gate >= 0.0f;
        const float exponential = expNonPositive(nonnegative ? -gate : gate);
        const float inverse = reciprocalOneToTwo(add(1.0f, exponential));
        const float sigmoid = nonnegative ? inverse : multiply(exponential, inverse);
        return multiply(multiply(gate, sigmoid), 1.0f);
    }

    /**
     * @brief Apply the canonical reciprocal to a widened positive half scale.
     * @param scale Scale published by the bounded QFN SwiGLU conversion.
     * @return The exact reciprocal word from the original Newton order.
     * @pre Scale is a positive, finite binary16 value widened to binary32.
     *      Its reciprocal and its widened representation are both normal in
     *      binary32, including when the original half value is subnormal.
     */
    __device__ __forceinline__ float qfn_swiglu_scale_inverse(float scale)
    {
        const uint32_t bits = __float_as_uint(scale);
        const float normalized = __uint_as_float(0x3f800000u | (bits & 0x007fffffu));
        const float inverse = llaminar2::device_fp32_contract::reciprocalOneToTwo(normalized);
        const uint32_t exponent_adjustment = 0x3f800000u - (bits & 0x7f800000u);
        return __uint_as_float(__float_as_uint(inverse) + exponent_adjustment);
    }

    /**
     * @brief Reduce split-K gate/up partials and quantize SwiGLU in one pass.
     *
     * Each lane sums k-partitions in the same order used by the former
     * reduce-then-quantize sequence, computes the same SwiGLU value, and writes
     * the same blockwise INT8 row layout for the down-projection kernel.  The
     * benefit is architectural: verifier graphs avoid a launch and a global
     * FP32 gate/up scratch round trip.
     * The launch admits the fixed split-K arithmetic policy, so publication
     * can issue independent partial loads with the original ordered FP32 fold.
     * The complete QFN warp uses an unsigned magnitude maximum and native
     * binary16 RNE over the canonical bounded scale range.
     * @tparam FixedColumns Exact admitted QFN pitch, or zero for runtime geometry.
     * @tparam ShareSigmoid Use the common rounded exponential chain on measured tiles.
     * @pre The QFN launch covers every column with complete warps and uses one
     *      block for each submitted original route and K32 output group.
     */
    template <int FixedColumns = 0, bool ShareSigmoid = false>
    __global__ void grouped_native_quantised_gate_up_ordered_kpart_reduce_swiglu_kernel(
        const float *__restrict__ gate_partials,
        const float *__restrict__ up_partials,
        const int *__restrict__ original_to_grouped,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int original_slot_base,
        int tile_route_slots,
        int N)
    {
        static_assert(!ShareSigmoid || FixedColumns > 0);
        constexpr int kTileN = 32;
        constexpr int kPartitions = llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        if constexpr (FixedColumns > 0)
        {
            static_assert(FixedColumns == kQFNDecodeGateUpN && FixedColumns % kTileN == 0);
            N = FixedColumns;
        }
        const int lane = threadIdx.x;
        const int block_idx = blockIdx.x;
        const int local_route = blockIdx.y;
        if constexpr (FixedColumns == 0)
        {
            if (local_route >= tile_route_slots)
                return;
        }

        const int grouped_slot =
            original_to_grouped[original_slot_base + local_route];
        if (grouped_slot < 0)
            return;

        const int n = block_idx * kTileN + lane;
        const bool active = FixedColumns > 0 || n < N;
        const size_t route_base =
            static_cast<size_t>(local_route) *
            static_cast<size_t>(kPartitions) * static_cast<size_t>(N);

        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        if (active)
        {
#pragma unroll
            for (int k_part = 0; k_part < kPartitions; ++k_part)
            {
                const size_t idx =
                    route_base + static_cast<size_t>(k_part) * static_cast<size_t>(N) + static_cast<size_t>(n);
                gate_sum += gate_partials[idx];
                up_sum += up_partials[idx];
            }
        }

        const float value = [&] {
            if constexpr (ShareSigmoid)
                return qfn_shared_sigmoid_silu(gate_sum) * up_sum;
            return active ? silu(gate_sum) * up_sum : 0.0f;
        }();
        using namespace llaminar2::device_q8_activation_contract;
        float scale;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if constexpr (FixedColumns > 0)
        {
            // Nonnegative FP32 words are ordered as unsigned integers.
            // The native maximum preserves zero and subnormal magnitudes.
            const uint32_t magnitude = __float_as_uint(value) & 0x7fffffffu;
            const float maximum = __uint_as_float(__reduce_max_sync(0xffffffffu, magnitude));
            const float bounded = boundedScale(maximum);
            // The clamp admits binary16 subnormals but excludes underflow,
            // overflow and signed zero. Native RNE publishes the same word.
            scale = __half2float(__float2half_rn(bounded));
        }
        else
#endif
        {
            float abs_value = maximumAbsoluteValue(value, 0.0f);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                abs_value = maximumAbsoluteValue(
                    abs_value, __shfl_xor_sync(0xffffffffu, abs_value, mask));
            scale = llaminar2::device_q8_activation_contract::scale(abs_value);
        }
        const int blocks_per_row = (N + kTileN - 1) / kTileN;
        if (lane == 0)
            swiglu_scales[static_cast<size_t>(grouped_slot) * static_cast<size_t>(blocks_per_row) +
                          static_cast<size_t>(block_idx)] = scale;

        if (active)
        {
            int32_t code;
            if constexpr (FixedColumns > 0)
                code = quantizePreparedInverse(value, qfn_swiglu_scale_inverse(scale));
            else
                code = quantize(value, scale);
            swiglu_int8[static_cast<size_t>(grouped_slot) * static_cast<size_t>(N) + static_cast<size_t>(n)] =
                static_cast<int8_t>(code);
        }
    }


    /**
     * @brief Produce one grouped split-K partial per original route slot.
     *
     * The grouped gate/up rows remain expert-major in scratch, but every block
     * resolves exactly one original `(token, route)` slot. Missing routes own
     * no partials: their publication kernel writes zero without reading this
     * scratch. That contract avoids writing and rereading an entire split-K
     * row for capacity-only or participant-remote slots in a retained graph.
     */
    template <uint8_t CodebookId>
    __global__ void grouped_prefill_down_canonical_kpart_scatter_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        float *__restrict__ partials,
        int original_slot_base,
        int tile_route_slots,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        const int k_part = blockIdx.y;
        const int local_route = blockIdx.z;
        if (local_route >= tile_route_slots ||
            k_part >= k_partitions || n >= N)
        {
            return;
        }

        const size_t partial_index =
            (static_cast<size_t>(local_route) *
                 static_cast<size_t>(k_partitions) +
             static_cast<size_t>(k_part)) *
                static_cast<size_t>(N) +
            static_cast<size_t>(n);
        const int original_slot = original_slot_base + local_route;
        const int grouped_slot = original_to_grouped[original_slot];
        if (grouped_slot < 0)
            return;

        const int expert_id = original_expert_ids[original_slot];
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
        const int8_t *slot_A =
            A_int8 + static_cast<size_t>(grouped_slot) * K;
        const float *slot_scales =
            scales_A_blockwise +
            static_cast<size_t>(grouped_slot) * blocks_per_row;
        partials[partial_index] =
            native_quantised_dot_desc_range_dispatch<CodebookId>(
                desc, n, slot_A, slot_scales, N, K, b_start, b_end);
    }

    /**
     * @brief Publish one narrow grouped Down row from its exact K32 partition tree.
     * @tparam CodebookId Physical reader chosen at admission or the mixed directory.
     * @tparam WideGrid Use the fixed-pitch reader for this captured node.
     * @tparam FixedReduction Logical K of that fixed-pitch reader.
     * @tparam TileColumns Number of output columns owned by each complete CTA.
     * @param activation_codes Graph-owned SwiGLU codes in compact grouped order.
     * @param activation_scales Graph-owned K32 scales in the same order.
     * @param descriptors Original published Down descriptors.
     * @param original_to_grouped Map from the public route slots to grouped activations.
     * @param original_experts Original logical IDs read from the same route plan.
     * @param grouped_weights Grouped factors, or null for raw cohort publication.
     * @param route_rows Public original-route rows, including zero for an absent route.
     * @param first_slot First original slot covered by the invocation.
     * @param N Full output width.
     * @param K Narrow admitted reduction width, with one K32 block per partition.
     * @param num_experts Exact original descriptor count.
     */
    template <uint8_t CodebookId, bool WideGrid, int FixedReduction = kWideVerifierDownK,
              int TileColumns = kVerifierDownTileColumns>
    __global__ void grouped_prefill_down_canonical_publish_k32_kernel(
        const int8_t *__restrict__ activation_codes,
        const float *__restrict__ activation_scales,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descriptors,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_experts,
        const float *__restrict__ grouped_weights,
        float *__restrict__ route_rows,
        int first_slot, int N, int K, int num_experts)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
        if constexpr (WideGrid)
        {
            N = kWideVerifierDownN;
            K = FixedReduction;
        }
#endif
        constexpr int kTileN = TileColumns;
        static_assert(kTileN == kVerifierDownTileColumns ||
                      (WideGrid && FixedReduction == kWideVerifierDownK && kWideVerifierDownN % kTileN == 0));
        const int column = blockIdx.x * kTileN + threadIdx.x;
        if (column >= N)
            return;
        const int original_slot = first_slot + blockIdx.y;
        const auto output = static_cast<size_t>(original_slot) * N + column;
        const int grouped_slot = original_to_grouped[original_slot];
        if (grouped_slot < 0)
        {
            route_rows[output] = 0.0f;
            return;
        }
        const int expert = original_experts[original_slot];
        const int blocks = K / 32;
        float sum = 0.0f;
        if (expert >= 0 && expert < num_experts)
        {
            const auto *codes = activation_codes + static_cast<size_t>(grouped_slot) * K;
            const auto *scales = activation_scales + static_cast<size_t>(grouped_slot) * blocks;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
            if constexpr (WideGrid)
                sum = native_quantised_moe_fixed_down<CodebookId, FixedReduction, kTileN>(
                    descriptors[expert], column, codes, scales);
            else
#endif
                sum = native_quantised_dot_desc_range_dispatch<CodebookId, NativeMoEDotOrder::VerifierK32>(
                    descriptors[expert], column, codes, scales, N, K, 0, blocks);
        }
        route_rows[output] = grouped_weights
            ? moe_weight_route_rn(grouped_weights[grouped_slot], sum) : sum;
    }

    /**
     * @brief Publish an SM86 short Q8 Down tile through the original ordered fold.
     *
     * The dispatch admits an exact eight-packet row. Each lane loads aligned
     * operands into two independent integer dot chains. Their exact integer
     * sum is scaled and added in the original FP32 K32 order.
     *
     * @param activation_codes Prepared Q8 SwiGLU rows in grouped order.
     * @param activation_scales Original FP32 K32 activation scales.
     * @param descriptors Published Down weight descriptors in expert ID order.
     * @param original_to_grouped Original-to-grouped route map.
     * @param original_experts Original expert ID for each route.
     * @param grouped_weights Optional original route weights in grouped order.
     * @param route_rows Output contributions in original route order.
     * @param first_slot First original slot covered by this launch.
     * @param column_count Admitted N in the weight-reference launch signature.
     * @param reduction_width Admitted K in the weight-reference launch signature.
     * @param num_experts Exact descriptor table length.
     * @pre The dispatcher admits the complete short Q8 rectangle and supplies
     *      its physical N and K in the standard captured argument positions.
     */
    __global__ void grouped_prefill_down_canonical_publish_k32_kernel_short_q8(
        const int8_t *__restrict__ activation_codes,
        const float *__restrict__ activation_scales,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descriptors,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_experts,
        const float *__restrict__ grouped_weights,
        float *__restrict__ route_rows,
        int first_slot, [[maybe_unused]] int column_count,
        [[maybe_unused]] int reduction_width, int num_experts)
    {
        constexpr int columns = kShortVerifierDownN;
        constexpr int packets = kShortVerifierDownK / 32;
        constexpr auto codebook = native_formats::Q8_0.codebook_id;
        static_assert(columns % kVerifierDownTileColumns == 0);
        const int column = blockIdx.x * kVerifierDownTileColumns + threadIdx.x;
        const int original = first_slot + blockIdx.y;
        const int grouped = original_to_grouped[original];
        const int expert = original_experts[original];
        float sum = 0.0f;
        if (grouped >= 0 && expert >= 0 && expert < num_experts)
        {
            const auto &desc = descriptors[expert];
            if (native_quantised_desc_shape_ok<codebook>(desc, columns, kShortVerifierDownK))
            {
                const auto row_packet = static_cast<size_t>(grouped) * packets;
                const auto *codes = reinterpret_cast<const int4 *>(activation_codes) + 2 * row_packet;
                const auto *weights = reinterpret_cast<const int4 *>(desc.payload) + 2 * column;
                const auto *weight_scales = static_cast<const uint16_t *>(desc.scales) + column;
                const auto *scales = activation_scales + row_packet;
#pragma unroll 2
                for (int packet = 0; packet < packets; ++packet)
                {
                    const int4 a0 = codes[packet * 2];
                    const int4 a1 = codes[packet * 2 + 1];
                    const int4 b0 = weights[packet * columns * 2];
                    const int4 b1 = weights[packet * columns * 2 + 1];
                    int low = __dp4a(a0.x, b0.x, 0);
                    int high = __dp4a(a1.x, b1.x, 0);
                    low = __dp4a(a0.y, b0.y, low);
                    high = __dp4a(a1.y, b1.y, high);
                    low = __dp4a(a0.z, b0.z, low);
                    high = __dp4a(a1.z, b1.z, high);
                    low = __dp4a(a0.w, b0.w, low);
                    high = __dp4a(a1.w, b1.w, high);
                    const float scale = llaminar2::cuda_native_quantised::fp16_bits_to_float(
                        weight_scales[packet * columns]);
                    const float block = llaminar2::device_native_quantised_contract::singleScaleBlock(
                        low + high, scale, scales[packet]);
                    sum = moe_accumulate_rn(sum, __fadd_rn(0.0f, block));
                }
            }
        }
        if (grouped >= 0 && grouped_weights)
            sum = moe_weight_route_rn(grouped_weights[grouped], sum);
        route_rows[static_cast<size_t>(original) * columns + column] = sum;
    }

    /**
     * @brief Reduce one expert row, then weight its persistent route slot.
     *
     * Applying the router weight after the ordered K fold is part of the
     * cross-backend arithmetic contract. Multiplying every partial separately
     * is algebraically similar but changes FP32 parenthesization.
     * Missing routes must be rejected before the scratch fold: their producer
     * intentionally leaves the previous transaction's partials untouched.
     */
    __global__ void grouped_prefill_down_canonical_kpart_reduce_kernel(
        const float *__restrict__ partials,
        const int *__restrict__ original_to_grouped,
        const float *__restrict__ grouped_weights,
        float *__restrict__ route_output,
        int original_slot_base,
        int tile_route_slots,
        int N,
        int k_partitions)
    {
        constexpr int kTileN = 64;
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int local_route = blockIdx.y;
        if (local_route >= tile_route_slots || n >= N)
            return;

        const int original_slot = original_slot_base + local_route;
        const int grouped_slot = original_to_grouped[original_slot];
        const size_t output_index =
            static_cast<size_t>(original_slot) * static_cast<size_t>(N) +
            static_cast<size_t>(n);
        if (grouped_slot < 0)
        {
            // This is public collective input, unlike private split-K scratch.
            // Every absent route must explicitly replace its old value by +0.
            route_output[output_index] = 0.0f;
            return;
        }

        const size_t partial_base =
            static_cast<size_t>(local_route) *
            static_cast<size_t>(k_partitions) * static_cast<size_t>(N);
        float sum = 0.0f;
        for (int k_part = 0; k_part < k_partitions; ++k_part)
        {
            sum = moe_accumulate_rn(
                sum,
                partials[
                    partial_base + static_cast<size_t>(k_part) *
                                       static_cast<size_t>(N) +
                    static_cast<size_t>(n)]);
        }
        route_output[output_index] = grouped_weights
            ? moe_weight_route_rn(grouped_weights[grouped_slot], sum) : sum;
    }

    /**
     * @brief Evaluate non-collective down routes without global split-K scratch.
     *
     * One eight-warp block owns 32 output columns for one verifier token. Each
     * warp evaluates one router route at a time. It preserves the canonical
     * arithmetic by summing K partitions in ascending order, applying one
     * router weight to the completed expert row, and then having warp zero sum
     * the completed routes in ascending router order.
     */
    template <uint8_t CodebookId>
    __global__ void grouped_prefill_down_canonical_kpart_fused_direct_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ descs,
        const int *__restrict__ original_to_grouped,
        const int *__restrict__ original_expert_ids,
        const float *__restrict__ grouped_weights,
        float *__restrict__ output,
        int original_slot_base,
        int token_base,
        int tile_rows,
        int top_k,
        int N,
        int K,
        int num_experts,
        int k_partitions)
    {
        constexpr int kWarpSize = 32;
        __shared__ float route_sums[kMaxTopK][kWarpSize];

        const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
        const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
        const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
        const int n = static_cast<int>(blockIdx.x) * kWarpSize + lane;
        const int local_token = blockIdx.y;
        if (local_token >= tile_rows)
            return;

#pragma unroll 1
        for (int route = warp; route < top_k; route += warps_per_block)
        {
            float route_sum = 0.0f;
            if (n < N)
            {
                const int local_route = local_token * top_k + route;
                const int original_slot = original_slot_base + local_route;
                const int grouped_slot = original_to_grouped[original_slot];
                const int expert_id = original_expert_ids[original_slot];
                if (grouped_slot >= 0 && expert_id >= 0 && expert_id < num_experts)
                {
                    const int blocks_per_row = K / 32;
                    const int blocks_per_part =
                        (blocks_per_row + k_partitions - 1) / k_partitions;
                    const DeviceNativeQuantisedMatrixDesc desc = descs[expert_id];
                    const int8_t *slot_A =
                        A_int8 + static_cast<size_t>(grouped_slot) * K;
                    const float *slot_scales =
                        scales_A_blockwise +
                        static_cast<size_t>(grouped_slot) * blocks_per_row;
                    const float route_weight = grouped_weights[grouped_slot];
                    float expert_sum = 0.0f;

#pragma unroll 1
                    for (int k_part = 0; k_part < k_partitions; ++k_part)
                    {
                        const int b_start = k_part * blocks_per_part;
                        const int b_end =
                            min(blocks_per_row, b_start + blocks_per_part);
                        float expert_partial = 0.0f;
                        if (b_start < b_end)
                        {
                            expert_partial =
                                native_quantised_dot_desc_range_dispatch<CodebookId>(
                                    desc, n, slot_A, slot_scales, N, K,
                                    b_start, b_end);
                        }
                        expert_sum =
                            moe_accumulate_rn(expert_sum, expert_partial);
                    }
                    route_sum = moe_weight_route_rn(route_weight, expert_sum);
                }
            }
            route_sums[route][lane] = route_sum;
        }
        __syncthreads();

        if (warp == 0 && n < N)
        {
            float output_sum = 0.0f;
#pragma unroll 1
            for (int route = 0; route < top_k; ++route)
            {
                output_sum =
                    moe_accumulate_rn(output_sum, route_sums[route][lane]);
            }
            output[
                static_cast<size_t>(token_base + local_token) *
                    static_cast<size_t>(N) +
                static_cast<size_t>(n)] = output_sum;
        }
    }



    /** Static compiler and occupancy evidence for one exact CUDA kernel. */
    struct GroupedPrefillKernelResources
    {
        int registers_per_thread = 0;
        size_t local_memory_bytes_per_thread = 0;
        size_t static_shared_memory_bytes = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;
    };

    /**
     * @brief Inspect one CUDA kernel at its exact production block width.
     *
     * This helper performs no allocation, launch, transfer, or synchronization.
     * `cudaFuncGetAttributes` exposes compiler-owned local memory, which is the
     * authoritative pre-timing spill gate; the occupancy query then reports the
     * number of resident blocks at the candidate's physical thread geometry.
     */
    template <typename Kernel>
    bool query_grouped_prefill_kernel_resources(
        Kernel kernel,
        int block_threads,
        GroupedPrefillKernelResources *resources)
    {
        if (!resources || block_threads <= 0)
            return false;

        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }
        if (block_threads > attributes.maxThreadsPerBlock)
            return false;

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                kernel,
                block_threads,
                0) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        *resources = GroupedPrefillKernelResources{
            .registers_per_thread = attributes.numRegs,
            .local_memory_bytes_per_thread = attributes.localSizeBytes,
            .static_shared_memory_bytes = attributes.sharedSizeBytes,
            .max_threads_per_block = attributes.maxThreadsPerBlock,
            .max_active_blocks_per_sm = active_blocks,
        };
        return active_blocks > 0;
    }

    /**
     * @brief Resolve the codebook-specialized producer used by one role.
     *
     * Component zero is ordered gate/up split-K production, component one is
     * non-collective direct down publication, and component two is canonical
     * route-major down production for collective execution. Common reduction
     * and gather kernels are queried separately because they are not codebook
     * specializations and must not be accidentally attributed to a candidate.
     */
    template <uint8_t CodebookId>
    bool query_grouped_prefill_codebook_component(
        int component,
        int block_threads,
        GroupedPrefillKernelResources *resources)
    {
        switch (component)
        {
        case 0:
            return query_grouped_prefill_kernel_resources(
                grouped_native_quantised_gate_up_ordered_kpart_scatter_kernel<CodebookId>,
                block_threads,
                resources);
        case 1:
            return query_grouped_prefill_kernel_resources(
                grouped_prefill_down_canonical_kpart_fused_direct_kernel<CodebookId>,
                block_threads,
                resources);
        case 2:
            return query_grouped_prefill_kernel_resources(
                grouped_prefill_down_canonical_kpart_scatter_kernel<CodebookId>,
                block_threads,
                resources);
        default:
            return false;
        }
    }

    int select_grouped_prefill_tile_m(int requested_tile_m, int max_tokens_per_expert)
    {
        switch (requested_tile_m)
        {
        case 2:
        case 4:
        case 8:
        case 16:
            return requested_tile_m;
        default:
            break;
        }

        // Tiny verifier groups are fastest with the compact two-row template
        // on CUDA. Wider runtime-M groups select larger tiles without changing
        // their row-wise arithmetic contract.
        if (max_tokens_per_expert <= 4)
            return 2;
        if (max_tokens_per_expert <= 8)
            return 8;
        return 16;
    }

} // namespace

extern "C"
{
    /**
     * @brief Inspect the production row-major Y/Z launch decomposition.
     *
     * This side-effect-free query gives focused regressions access to the exact
     * geometry used by every large-M MoE launcher. It performs no CUDA runtime
     * operation, allocation, transfer, launch, or synchronization.
     *
     * @param column_blocks Positive X-dimension column tile count.
     * @param logical_rows Positive semantic row count.
     * @param grid_x Receives the physical X dimension.
     * @param grid_y Receives the bounded physical Y dimension.
     * @param grid_z Receives the Y-overflow plane count.
     * @return true when the requested positive geometry is representable.
     */
    bool cudaMoE_query_row_major_grid(
        unsigned int column_blocks,
        int logical_rows,
        unsigned int *grid_x,
        unsigned int *grid_y,
        unsigned int *grid_z)
    {
        if (column_blocks == 0 || logical_rows <= 0 ||
            !grid_x || !grid_y || !grid_z)
        {
            return false;
        }

        const dim3 grid = make_row_major_grid(column_blocks, logical_rows);
        *grid_x = grid.x;
        *grid_y = grid.y;
        *grid_z = grid.z;
        return grid.x == column_blocks &&
               grid.y > 0 && grid.y <= kCudaGridYDimensionLimit &&
               grid.z > 0 && grid.z <= kCudaGridYDimensionLimit;
    }

    /**
     * @brief Query static resources for one production grouped-prefill kernel.
     *
     * The query is deliberately side-effect free: it performs no allocation,
     * launch, copy, or synchronization. Candidate components are inspected at
     * their exact physical block width so a trainer can reject compiler spills
     * before graph capture and can attach honest occupancy evidence to the
     * candidate it eventually times.
     *
     * Component identities are stable and intentionally explicit:
     * - 0: codebook-specialized ordered gate/up split-K producer;
     * - 1: codebook-specialized non-collective direct down producer;
     * - 2: codebook-specialized canonical-route down split-K producer;
     * - 3: original-token hidden-row blockwise quantizer;
     * - 4: common gate/up split-K reduction and SwiGLU quantizer;
     * - 5: common canonical-route down split-K reduction.
     *
     * @param codebook_id NativeQuantised execution codebook for components 0..2.
     * @param component Stable component identity described above.
     * @param block_threads Exact production block width in CUDA threads.
     * @param registers_per_thread Receives compiler-assigned registers.
     * @param local_memory_bytes_per_thread Receives compiler local scratch;
     *        any non-zero value disqualifies a tuning candidate.
     * @param static_shared_memory_bytes Receives static shared memory bytes.
     * @param max_threads_per_block Receives the compiled launch ceiling.
     * @param max_active_blocks_per_sm Receives occupancy at `block_threads`.
     * @return true only when the exact compiled kernel can be inspected.
     */
    bool cudaMoE_grouped_prefill_query_kernel_resources(
        uint8_t codebook_id,
        int component,
        int block_threads,
        int *registers_per_thread,
        size_t *local_memory_bytes_per_thread,
        size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
        if (!registers_per_thread || !local_memory_bytes_per_thread ||
            !static_shared_memory_bytes || !max_threads_per_block ||
            !max_active_blocks_per_sm)
        {
            return false;
        }

        GroupedPrefillKernelResources resources{};
        bool queried = false;
        switch (component)
        {
        case 3:
            queried = block_threads == kWarpsPerQuantBlock * 32 &&
                      query_grouped_prefill_kernel_resources(
                          moe_prefill_hidden_quantize_rows_kernel,
                          block_threads,
                          &resources);
            break;
        case 4:
            queried = block_threads == 32 &&
                      query_grouped_prefill_kernel_resources(
                          grouped_native_quantised_gate_up_ordered_kpart_reduce_swiglu_kernel<>,
                          block_threads,
                          &resources);
            break;
        case 5:
            queried = block_threads == 64 &&
                      query_grouped_prefill_kernel_resources(
                          grouped_prefill_down_canonical_kpart_reduce_kernel,
                          block_threads,
                          &resources);
            break;
        default:
            break;
        }

        if (component >= 0 && component <= 2)
        {
            const bool valid_width =
                ((component == 0 || component == 2) &&
                 block_threads >= 64 && block_threads <= 256 &&
                 (block_threads % 32) == 0) ||
                (component == 1 && block_threads >= 256 &&
                 block_threads <= 512 && (block_threads % 32) == 0);
            if (!valid_width)
                return false;

#define QUERY_GROUPED_PREFILL_CODEBOOK(CB)                                      \
    case CB:                                                                    \
        queried = query_grouped_prefill_codebook_component<CB>(                 \
            component, block_threads, &resources);                              \
        break

            switch (codebook_id)
            {
                QUERY_GROUPED_PREFILL_CODEBOOK(0);
                QUERY_GROUPED_PREFILL_CODEBOOK(4);
                QUERY_GROUPED_PREFILL_CODEBOOK(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook);
                QUERY_GROUPED_PREFILL_CODEBOOK(5);
                QUERY_GROUPED_PREFILL_CODEBOOK(6);
                QUERY_GROUPED_PREFILL_CODEBOOK(7);
                QUERY_GROUPED_PREFILL_CODEBOOK(8);
                QUERY_GROUPED_PREFILL_CODEBOOK(9);
                QUERY_GROUPED_PREFILL_CODEBOOK(10);
                QUERY_GROUPED_PREFILL_CODEBOOK(11);
                QUERY_GROUPED_PREFILL_CODEBOOK(12);
                QUERY_GROUPED_PREFILL_CODEBOOK(13);
                QUERY_GROUPED_PREFILL_CODEBOOK(14);
                QUERY_GROUPED_PREFILL_CODEBOOK(15);
                QUERY_GROUPED_PREFILL_CODEBOOK(16);
                QUERY_GROUPED_PREFILL_CODEBOOK(17);
                QUERY_GROUPED_PREFILL_CODEBOOK(19);
                QUERY_GROUPED_PREFILL_CODEBOOK(
                    llaminar2::kNativeQuantisedExpandedInt8MinCodebook);
            case kMixedCodebookSentinel:
                if (component == 1 || component == 2)
                {
                    queried = query_grouped_prefill_codebook_component<
                        kMixedCodebookSentinel>(
                        component, block_threads, &resources);
                }
                break;
            default:
                return false;
            }

#undef QUERY_GROUPED_PREFILL_CODEBOOK
        }

        if (!queried)
            return false;
        *registers_per_thread = resources.registers_per_thread;
        *local_memory_bytes_per_thread =
            resources.local_memory_bytes_per_thread;
        *static_shared_memory_bytes = resources.static_shared_memory_bytes;
        *max_threads_per_block = resources.max_threads_per_block;
        *max_active_blocks_per_sm = resources.max_active_blocks_per_sm;
        return true;
    }


    /**
     * @brief Execute the complete device-resident CUDA routed-MoE prefill path.
     *
     * The caller supplies persistent graph-owned tensors and scratch. Gate/up
     * and down arithmetic use the fixed partition counts in
     * `CUDAMoEBatchInvariantPolicy`; only the gate/up producer block width is a
     * capture-time dispatch choice. Changing that width redistributes output
     * columns without changing the per-element reduction order, which is why an
     * authenticated exact overlay may tune it while partition counts may not.
     *
     * @param d_hidden FP32 token rows resident on `device_idx`.
     * @param d_prequantized_hidden Optional compatible Q8 activation rows.
     * @param d_prequantized_hidden_scales Scales paired with prequantized rows.
     * @param d_gate_desc_table Device descriptor table for expert gate weights.
     * @param d_up_desc_table Device descriptor table for expert up weights.
     * @param d_down_desc_table Device descriptor table for expert down weights.
     * @param d_group_counts Per-expert route counts published by the planner.
     * @param d_group_offsets Exclusive offsets into grouped route storage.
     * @param d_group_token_indices Grouped token or route-slot indices.
     * @param d_original_to_grouped Original-route to grouped-route mapping.
     * @param d_original_expert_ids Expert id for every original route slot.
     * @param d_active_expert_ids Compact active-expert ids for launch grids.
     * @param d_group_weights Router weights in grouped route order.
     * @param d_scratch_A_int8 Persistent gathered Q8 activation payload.
     * @param d_scratch_scales Persistent gathered activation scales.
     * @param d_scratch_gate Persistent direct-path gate output scratch.
     * @param d_scratch_up Persistent direct-path up output scratch.
     * @param d_gate_partials Ordered gate split-K partials.
     * @param d_up_partials Ordered up split-K partials.
     * @param d_scratch_swiglu_int8 Persistent fused SwiGLU Q8 payload.
     * @param d_scratch_swiglu_scales Persistent fused SwiGLU scales.
     * @param d_down_partials Ordered down split-K partials.
     * @param d_scratch_down_out Persistent grouped down-output scratch.
     * @param d_output Final token-major output, or null for canonical publish.
     * @param d_canonical_route_contributions Optional route-major publication.
     * @param num_experts Total logical routed experts.
     * @param d_model Hidden width in scalar elements.
     * @param intermediate Routed expert intermediate width.
     * @param max_tokens_per_expert Planner-owned grouped-row capacity.
     * @param total_slots Exact `M * top_k` route-slot count.
     * @param top_k Number of selected experts per token.
     * @param active_expert_slots Number of compact active expert groups.
     * @param grouped_indices_are_route_slots Whether grouped indices name slots.
     * @param gateup_codebook_id Uniform gate/up codebook or mixed sentinel.
     * @param down_codebook_id Uniform down codebook or mixed sentinel.
     * @param gateup_codebook_mask Every gate/up codebook present in the table.
     * @param down_codebook_mask Every down codebook present in the table.
     * @param gateup_k_partitions Fixed byte-equivalent gate/up partition count.
     * @param gateup_ordered_tile_n Capture-time gate/up producer block width.
     * @param down_k_partitions Fixed byte-equivalent down partition count.
     * @param splitk_tile_rows Persistent partial-scratch row tile capacity.
     * @param projection_engine Capture-time grouped projection implementation.
     * @param imma_gateup_columns Output columns owned by one fused IMMA CTA.
     * @param imma_down_columns Output columns owned by one down IMMA CTA.
     * @param imma_gateup_schedule Projection-to-warp ownership in gate/up.
     * @param device_idx CUDA ordinal that owns every supplied pointer.
     * @param stream Exact non-null producer stream embedded by graph capture.
     * @param execution Phase and full-source/local-column arithmetic identity.
     * @param gate_contract Validation contract borrowed from the ordered runtime publisher.
     * @return true after all launches have been enqueued without launch errors.
     */
    bool cudaMoE_grouped_prefill_pipeline(
        const float *d_hidden,
        const int8_t *d_prequantized_hidden,
        const float *d_prequantized_hidden_scales,
        const DeviceNativeQuantisedMatrixDesc *d_gate_desc_table,
        const DeviceNativeQuantisedMatrixDesc *d_up_desc_table,
        const DeviceNativeQuantisedMatrixDesc *d_down_desc_table,
        const int *d_group_counts,
        const int *d_group_offsets,
        uint32_t *d_group_work_directory,
        int group_work_directory_entries,
        const int *d_group_token_indices,
        const int *d_original_to_grouped,
        const int *d_original_expert_ids,
        const int *d_active_expert_ids,
        const float *d_group_weights,
        int8_t *d_scratch_A_int8,
        float *d_scratch_scales,
        float *d_scratch_gate,
        float *d_scratch_up,
        float *d_gate_partials,
        float *d_up_partials,
        int8_t *d_scratch_swiglu_int8,
        float *d_scratch_swiglu_scales,
        float *d_down_partials,
        float *d_scratch_down_out,
        float *d_output,
        float *d_canonical_route_contributions,
        int num_experts,
        int d_model,
        int intermediate,
        int max_tokens_per_expert,
        int total_slots,
        int top_k,
        int active_expert_slots,
        int grouped_indices_are_route_slots,
        uint8_t gateup_codebook_id,
        uint8_t down_codebook_id,
        uint32_t gateup_codebook_mask,
        uint32_t down_codebook_mask,
        int gateup_k_partitions,
        int gateup_ordered_tile_n,
        int down_k_partitions,
        int splitk_tile_rows,
        llaminar2::CUDAMoEBatchInvariantPolicy::GroupedProjectionEngine
            projection_engine,
        llaminar2::cuda::moe::GroupedImmaColumns imma_gateup_columns,
        llaminar2::cuda::moe::GroupedImmaColumns imma_down_columns,
        llaminar2::cuda::moe::GroupedImmaGateUpSchedule
            imma_gateup_schedule,
        int device_idx,
        void *stream,
        const llaminar2::MoEPrefillProjectionExecution &execution,
        llaminar2::CUDAMoERuntimeGateContract gate_contract)
    {
        // One LUT family producer handles both factor programs; each actual
        // descriptor remains untouched and selects its own source scale.
        gateup_codebook_mask = llaminar2::cuda_native_quantised::native_quantised_iq4_launch_mask(gateup_codebook_mask);
        down_codebook_mask = llaminar2::cuda_native_quantised::native_quantised_iq4_launch_mask(down_codebook_mask);
        const bool executes_gateup = execution.executesGateUp();
        const bool executes_down = execution.executesDown();
        const int down_columns = execution.columnCount();
        if (!d_group_counts || !d_group_offsets ||
            !d_scratch_swiglu_int8 || !d_scratch_swiglu_scales ||
            (executes_gateup && (!d_hidden || !d_gate_desc_table || !d_up_desc_table ||
                !d_group_token_indices || !d_scratch_A_int8 || !d_scratch_scales ||
                !d_scratch_gate || !d_scratch_up)) ||
            (executes_down && (!d_down_desc_table || !d_group_weights || !d_scratch_down_out ||
                (!d_output && !d_canonical_route_contributions))) ||
            execution.modelColumns() != d_model ||
            num_experts <= 0 || d_model <= 0 || intermediate <= 0 ||
            max_tokens_per_expert <= 0 || total_slots <= 0 || !stream ||
            top_k <= 0 || top_k > kMaxTopK || (total_slots % top_k) != 0 ||
            active_expert_slots < 0 ||
            (d_model % 32) != 0 || (intermediate % 32) != 0)
        {
            std::fprintf(stderr, "[cudaMoE_grouped_prefill_pipeline] invalid arguments\n");
            return false;
        }
        if ((d_prequantized_hidden == nullptr) != (d_prequantized_hidden_scales == nullptr) ||
            (grouped_indices_are_route_slots != 0 && grouped_indices_are_route_slots != 1))
            return false;
        using GroupedProjectionEngine =
            llaminar2::CUDAMoEBatchInvariantPolicy::GroupedProjectionEngine;
        const bool use_grouped_imma =
            projection_engine ==
            GroupedProjectionEngine::TensorCoreImmaPrefill;
        if (projection_engine !=
                GroupedProjectionEngine::OrderedDp4aVerifier &&
            !use_grouped_imma)
        {
            return false;
        }
        if (use_grouped_imma &&
            (!d_group_work_directory || group_work_directory_entries <= 0))
        {
            return false;
        }
        if (use_grouped_imma &&
            (!llaminar2::cuda::moe::validGroupedImmaGateUpColumns(
                 imma_gateup_columns) ||
             !llaminar2::cuda::moe::validGroupedImmaDownColumns(
                 imma_down_columns) ||
             !llaminar2::cuda::moe::validGroupedImmaGateUpSchedule(
                 imma_gateup_schedule)))
        {
            return false;
        }
        const bool use_active_expert_grid = active_expert_slots > 0;
        const bool valid_gateup_k_partitions =
            gateup_k_partitions ==
            llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
        const bool gateup_kpart_requested = gateup_k_partitions > 0;
        const bool valid_gateup_ordered_tile_n =
            gateup_ordered_tile_n >= 64 && gateup_ordered_tile_n <= 256 &&
            (gateup_ordered_tile_n % 32) == 0;
        const bool use_gateup_kpart =
            !use_grouped_imma && use_active_expert_grid && gateup_kpart_requested &&
            valid_gateup_k_partitions && d_gate_partials && d_up_partials &&
            d_original_to_grouped && d_original_expert_ids;
        const bool valid_down_k_partitions =
            down_k_partitions ==
            llaminar2::CUDAMoEBatchInvariantPolicy::down_k_partitions;
        const bool down_kpart_requested = down_k_partitions > 0;
        const bool canonical_publication =
            d_canonical_route_contributions != nullptr;
        const bool narrow_canonical_down = canonical_publication && intermediate / 32 <= down_k_partitions;
        const bool use_ordered_down_kpart =
            !use_grouped_imma && use_active_expert_grid && down_kpart_requested &&
            d_original_to_grouped &&
            d_original_expert_ids &&
            (!canonical_publication || narrow_canonical_down || d_down_partials) &&
            valid_down_k_partitions;
        const bool raw_cohort =
            execution.routeWeighting() == llaminar2::MoEExpertRouteWeighting::CompletedCohort;
        if (raw_cohort && (!canonical_publication || d_output || (!use_grouped_imma && !use_ordered_down_kpart)))
            return false;
        const float *const projection_weights = raw_cohort ? nullptr : d_group_weights;
        if (executes_gateup && gateup_kpart_requested &&
            (!use_gateup_kpart || !valid_gateup_ordered_tile_n))
            return false;
        if (down_k_partitions > 0 && !valid_down_k_partitions)
            return false;
        if (executes_down && down_kpart_requested && !use_ordered_down_kpart)
            return false;
        if ((use_gateup_kpart || use_ordered_down_kpart) && splitk_tile_rows <= 0)
            return false;
        if (use_grouped_imma &&
            (gateup_kpart_requested || down_kpart_requested))
        {
            return false;
        }
        if (use_active_expert_grid &&
            (!d_active_expert_ids ||
             active_expert_slots > total_slots ||
             active_expert_slots > num_experts))
            return false;
        if (use_grouped_imma && use_active_expert_grid &&
            !d_original_to_grouped)
            return false;
        const int expert_grid = use_active_expert_grid ? active_expert_slots : num_experts;
        const int seq_len = total_slots / top_k;
        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        /*
         * A rank with no local routes has already received an overwrite-only
         * zero publication from its owner. No projection is semantically live,
         * so capturing empty rectangular GEMMs would only burn replay time and
         * expose stale scratch. This is an explicit no-work state, not another
         * execution implementation.
         */
        if (!use_active_expert_grid)
            return true;

        if (executes_gateup)
        {
            const auto source_rows = grouped_indices_are_route_slots
                ? llaminar2::MoEGroupedSourceRows::routeIndices(d_group_token_indices, seq_len, top_k)
                : llaminar2::MoEGroupedSourceRows::tokenIndices(d_group_token_indices, seq_len);
            const auto *gateup_input = d_prequantized_hidden ? d_prequantized_hidden : d_scratch_A_int8;
            const auto *gateup_scales = d_prequantized_hidden_scales ? d_prequantized_hidden_scales : d_scratch_scales;
            // The router publication is already ordered before this stage. If
            // absent, the same canonical producer writes one row per token.
            // Neither case creates a top-k-expanded activation representation.
            if (!d_prequantized_hidden)
            {
                // One warp per 32-col quant block; pack kWarpsPerQuantBlock warps per CUDA
                // block so each block covers kWarpsPerQuantBlock*32 columns. grid.x rounds up.
                const int blocks_per_row = d_model / 32;
                const dim3 grid = make_row_major_grid(
                    static_cast<unsigned int>(
                        (blocks_per_row + kWarpsPerQuantBlock - 1) /
                        kWarpsPerQuantBlock),
                    seq_len);
                dim3 block(kWarpsPerQuantBlock * 32);
                moe_prefill_hidden_quantize_rows_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_hidden, d_scratch_A_int8, d_scratch_scales, seq_len, d_model);
                if (!finishGroupedPrefillLaunch("cudaMoE_prefill_hidden_quantize_rows", cuda_stream))
                    return false;
            }

            if (use_grouped_imma)
            {
                if (!cudaMoEGroupedImma_buildDirectory(
                        d_group_counts,
                        d_group_work_directory,
                        num_experts,
                        total_slots,
                        group_work_directory_entries,
                        device_idx,
                        stream))
                {
                    return false;
                }

                const auto *gate_descriptors = reinterpret_cast<
                    const llaminar2::DeviceNativeQuantisedMatrixDesc *>(
                        d_gate_desc_table);
                const auto *up_descriptors = reinterpret_cast<
                    const llaminar2::DeviceNativeQuantisedMatrixDesc *>(
                        d_up_desc_table);
                if (!cudaMoEGroupedImma_projectGateUpSwiGlu(
                        gateup_input,
                        gateup_scales,
                        source_rows,
                        gate_descriptors,
                        up_descriptors,
                        d_group_counts,
                        d_group_offsets,
                        d_group_work_directory,
                        d_scratch_swiglu_int8,
                        d_scratch_swiglu_scales,
                        group_work_directory_entries,
                        num_experts,
                        total_slots,
                        intermediate,
                        d_model,
                        gateup_codebook_mask,
                        llaminar2::CUDAMoEBatchInvariantPolicy::
                            gate_up_k_partitions,
                        imma_gateup_columns,
                        imma_gateup_schedule,
                        device_idx,
                        stream))
                {
                    return false;
                }
            }
            else if (use_gateup_kpart)
            {
                const int kTileN = gateup_ordered_tile_n;
                constexpr int kReduceTileN = 32;
                dim3 block(kTileN);
                dim3 reduce_block(kReduceTileN);

                const bool fixed_partitions =
                    gateup_k_partitions == llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
                const bool qfn_gate_geometry = intermediate == kQFNDecodeGateUpN && d_model == kQFNDecodeGateUpK;
                const bool use_fixed_qfn = fixed_partitions && qfn_gate_geometry &&
                    (kTileN == 64 || kTileN == 128) && is_sm86_cuda_device(device_idx);
                const bool fixed_gate_geometry = kTileN == 128 && (use_fixed_qfn ||
                    (fixed_partitions && intermediate == kWideVerifierGateN &&
                     (d_model == 512 || d_model == kWideVerifierGateK) && is_sm86_cuda_device(device_idx)));

                for (int token_base = 0; token_base < seq_len; token_base += splitk_tile_rows)
                {
                    const int tile_rows = std::min(splitk_tile_rows, seq_len - token_base);
                    const int original_slot_base = token_base * top_k;
                    const int tile_route_slots = tile_rows * top_k;
                    dim3 scatter_grid(
                        (intermediate + kTileN - 1) / kTileN,
                        tile_route_slots,
                        gateup_k_partitions);

                    const auto launch_gate = [&]<uint8_t CodebookId>() {
                        using Traits = llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;
                        constexpr auto packed = Traits::is_iq_grid
                            ? NativeMoEDotOrder::VerifierWideGrid : NativeMoEDotOrder::VerifierK32;
                        constexpr auto unrolled = NativeMoEDotOrder::PartitionPartial;
                        const auto issue = [&]<int Pitch, int Packets, NativeMoEDotOrder Order,
                                               int ExactTile = 0, unsigned RoutesPerToken = 0, bool PlanValidated = false>() {
                            constexpr auto kernel = [] {
                                if constexpr (PlanValidated && kQFNBoundedGateRegisters<CodebookId>)
                                    return grouped_native_quantised_gate_up_ordered_kpart_scatter_kernel_bounded<
                                        CodebookId, Pitch, Packets, Order, ExactTile, RoutesPerToken, PlanValidated>;
                                else
                                    return grouped_native_quantised_gate_up_ordered_kpart_scatter_kernel<
                                        CodebookId, Pitch, Packets, Order, ExactTile, RoutesPerToken, PlanValidated>;
                            }();
                            kernel<<<scatter_grid, block, 0, cuda_stream>>>(
                                    gateup_input, gateup_scales, source_rows, d_gate_desc_table, d_up_desc_table,
                                    d_original_to_grouped, d_original_expert_ids, d_gate_partials, d_up_partials,
                                    original_slot_base, tile_route_slots, intermediate, d_model, num_experts,
                                    gateup_k_partitions);
                        };
                        if constexpr (llaminar2::CUDAMoEBatchInvariantPolicy::qfnPackedGateUp(CodebookId))
                        {
                            if (use_fixed_qfn)
                            {
                                if constexpr (Traits::is_iq_grid && CodebookId != native_formats::IQ3_S.codebook_id)
                                {
                                    if (kTileN == 64)
                                    {
                                        const auto exact = [&]<unsigned RoutesPerToken>() {
                                            if (gate_contract.enabled() && gate_contract.codebook() == CodebookId)
                                                return issue.template operator()<kQFNDecodeGateUpN,
                                                    kQFNDecodeGateUpPackets, packed, 64, RoutesPerToken, true>();
                                            return issue.template operator()<kQFNDecodeGateUpN,
                                                kQFNDecodeGateUpPackets, packed, 64, RoutesPerToken>();
                                        };
                                        if constexpr (CodebookId == native_formats::IQ1_S.codebook_id ||
                                                      CodebookId == native_formats::IQ2_XXS.codebook_id)
                                        {
                                            if (seq_len == llaminar2::CUDAMoEBatchInvariantPolicy::qfn_verifier_gate_plan_rows &&
                                                static_cast<unsigned>(top_k) == kQFNVerifierRoutesPerToken &&
                                                grouped_indices_are_route_slots)
                                                return exact.template operator()<kQFNVerifierRoutesPerToken>();
                                        }
                                        return exact.template operator()<0>();
                                    }
                                }
                                return issue.template operator()<kQFNDecodeGateUpN, kQFNDecodeGateUpPackets, packed>();
                            }
                        }
                        if constexpr (Traits::is_iq4_lut)
                        {
                            if (use_fixed_qfn && kTileN == 128)
                                return issue.template operator()<kQFNDecodeGateUpN, kQFNDecodeGateUpPackets, unrolled>();
                            if (fixed_gate_geometry && intermediate == kWideVerifierGateN && d_model == kWideVerifierGateK)
                                return issue.template operator()<kWideVerifierGateN, kWideVerifierGatePackets, unrolled>();
                        }
                        else
                        {
                            if (fixed_gate_geometry)
                            {
                                if (intermediate == kWideVerifierGateN && d_model == 512)
                                    return issue.template operator()<kWideVerifierGateN, 1, packed>();
                                if (intermediate == kWideVerifierGateN && d_model == kWideVerifierGateK)
                                {
                                    constexpr bool prefer_unrolled =
                                        CodebookId == native_formats::Q4_0.codebook_id ||
                                        CodebookId == native_formats::Q4_K.codebook_id ||
                                        CodebookId == native_formats::Q5_1.codebook_id ||
                                        CodebookId == native_formats::Q5_K.codebook_id;
                                    return issue.template operator()<
                                        kWideVerifierGateN, kWideVerifierGatePackets,
                                        prefer_unrolled ? unrolled : packed>();
                                }
                                if constexpr (CodebookId == native_formats::Q4_K.codebook_id ||
                                              CodebookId == native_formats::Q5_K.codebook_id ||
                                              CodebookId == native_formats::Q5_1.codebook_id)
                                {
                                    if (intermediate == kQFNDecodeGateUpN && d_model == kQFNDecodeGateUpK)
                                        return issue.template operator()<
                                            kQFNDecodeGateUpN, kQFNDecodeGateUpPackets, unrolled>();
                                }
                            }
                        }
                        issue.template operator()<0, 0, unrolled>();
                    };

    #define LAUNCH_GROUPED_GATEUP_ORDERED_KPART(CB) launch_gate.template operator()<CB>()

                    bool launched_gateup = false;
    #define LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(CB)                                        \
        do {                                                                                           \
            if (gateup_codebook_mask & (uint32_t{1} << (CB))) {                                       \
                LAUNCH_GROUPED_GATEUP_ORDERED_KPART(CB);                                               \
                if (!finishGroupedPrefillLaunch(                                                       \
                        "cudaMoE_grouped_gate_up_ordered_kpart_prefill", cuda_stream))                \
                    return false;                                                                      \
                launched_gateup = true;                                                                \
            }                                                                                          \
        } while (0)

                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(0);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(4);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(5);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(6);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(7);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(8);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(9);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(10);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(11);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(12);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(13);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(14);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(15);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(16);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(17);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(19);
                    LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT(
                        llaminar2::kNativeQuantisedExpandedInt8MinCodebook);
                    if (!launched_gateup)
                    {
                        std::fprintf(
                            stderr,
                            "[cudaMoE_grouped_prefill_pipeline] unsupported ordered kpart gate/up codebook_id=%u mask=0x%x\n",
                            static_cast<unsigned>(gateup_codebook_id),
                            static_cast<unsigned>(gateup_codebook_mask));
                        return false;
                    }

                    dim3 reduce_grid(
                        (intermediate + kReduceTileN - 1) / kReduceTileN,
                        tile_route_slots);
                    const auto publish = [&]<int FixedColumns, bool ShareSigmoid = false>() {
                        grouped_native_quantised_gate_up_ordered_kpart_reduce_swiglu_kernel<FixedColumns, ShareSigmoid>
                            <<<reduce_grid, reduce_block, 0, cuda_stream>>>(
                                d_gate_partials, d_up_partials, d_original_to_grouped,
                                d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                                original_slot_base, tile_route_slots, intermediate);
                    };
                    if (use_fixed_qfn && kTileN == 64 && gate_contract.enabled())
                        publish.template operator()<kQFNDecodeGateUpN, true>();
                    else if (use_fixed_qfn)
                        publish.template operator()<kQFNDecodeGateUpN>();
                    else
                        publish.template operator()<0>();
                    if (!finishGroupedPrefillLaunch(
                            "cudaMoE_grouped_gate_up_ordered_kpart_reduce_swiglu",
                            cuda_stream))
                    {
                        return false;
                    }

    #undef LAUNCH_GROUPED_GATEUP_ORDERED_KPART_IF_PRESENT
    #undef LAUNCH_GROUPED_GATEUP_ORDERED_KPART
                }
            }
            else
            {
                const int requestedTileM =
                    llaminar2::debugEnv().gemm.cuda_moe_prefill_tile_m;
                const int kTileM =
                    select_grouped_prefill_tile_m(requestedTileM, max_tokens_per_expert);
                constexpr int kTileN = 128;
                const bool fuse_swiglu =
                    llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu;
                dim3 grid(
                    (intermediate + kTileN - 1) / kTileN,
                    (max_tokens_per_expert + kTileM - 1) / kTileM,
                    expert_grid);
                dim3 block(kTileN);

    #define LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, TM)                                                  \
        grouped_native_quantised_gate_up_prefill_kernel<CB, TM, kTileN>                                    \
            <<<grid, block, 0, cuda_stream>>>(                                                         \
                gateup_input, gateup_scales, source_rows, d_gate_desc_table, d_up_desc_table,          \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,             \
                d_scratch_gate, d_scratch_up, intermediate, d_model)
    #define LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, TM)                                           \
        grouped_native_quantised_gate_up_swiglu_prefill_kernel<CB, TM, kTileN>                             \
            <<<grid, block, 0, cuda_stream>>>(                                                         \
                gateup_input, gateup_scales, source_rows, d_gate_desc_table, d_up_desc_table,          \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,             \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, intermediate, d_model)
    #define LAUNCH_GROUPED_GATEUP_PREFILL(CB)                                                         \
        do {                                                                                          \
            if (fuse_swiglu) {                                                                        \
                if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 16);               \
                else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 8);                \
                else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 4);                \
                else                   LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM(CB, 2);                \
            } else {                                                                                  \
                if (kTileM == 16)      LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 16);                      \
                else if (kTileM == 8)  LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 8);                       \
                else if (kTileM == 4)  LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 4);                       \
                else                   LAUNCH_GROUPED_GATEUP_PREFILL_TM(CB, 2);                       \
            }                                                                                         \
        } while (0)

                bool launched_gateup = false;
    #define LAUNCH_GROUPED_GATEUP_IF_PRESENT(CB)                                                      \
        do {                                                                                          \
            if (gateup_codebook_mask & (uint32_t{1} << (CB))) {                                      \
                LAUNCH_GROUPED_GATEUP_PREFILL(CB);                                                    \
                if (!finishGroupedPrefillLaunch("cudaMoE_grouped_gate_up_prefill", cuda_stream))     \
                    return false;                                                                     \
                launched_gateup = true;                                                               \
            }                                                                                         \
        } while (0)

                LAUNCH_GROUPED_GATEUP_IF_PRESENT(0);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(4);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(5);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(6);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(7);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(8);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(9);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(10);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(11);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(12);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(13);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(14);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(15);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(16);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(17);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(19);
                LAUNCH_GROUPED_GATEUP_IF_PRESENT(
                    llaminar2::kNativeQuantisedExpandedInt8MinCodebook);
                if (!launched_gateup)
                {
                    std::fprintf(
                        stderr,
                        "[cudaMoE_grouped_prefill_pipeline] unsupported gate/up codebook_id=%u mask=0x%x\n",
                        static_cast<unsigned>(gateup_codebook_id),
                        static_cast<unsigned>(gateup_codebook_mask));
                    return false;
                }

    #undef LAUNCH_GROUPED_GATEUP_IF_PRESENT
    #undef LAUNCH_GROUPED_GATEUP_PREFILL
    #undef LAUNCH_GROUPED_GATEUP_SWIGLU_PREFILL_TM
    #undef LAUNCH_GROUPED_GATEUP_PREFILL_TM
            }

            // Separate SwiGLU + blockwise-quant pass. Both grouped-IMMA and the
            // direct fused gate/up kernel publish d_scratch_swiglu_int8/scales in
            // their epilogues, so only the unfused ordered-DP4A geometry enters
            // this launch.
            if (!use_grouped_imma &&
                !use_gateup_kpart &&
                !llaminar2::debugEnv().gemm.cuda_moe_prefill_fuse_swiglu)
            {
                const dim3 grid = make_row_major_grid(
                    static_cast<unsigned int>(intermediate / 32),
                    total_slots);
                dim3 block(32);
                grouped_prefill_swiglu_quantize_blockwise_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_scratch_gate, d_scratch_up,
                    d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                    total_slots, intermediate);
                if (!finishGroupedPrefillLaunch("cudaMoE_grouped_swiglu_quantize_prefill", cuda_stream))
                    return false;
            }
        }
        if (!executes_down)
            return true;

        // A separately recorded down phase owns its complete consumer grouping,
        // which may differ from the owner-only gate/up grouping before exchange.
        // Rebuild only that phase's directory; no host observes live route counts.
        if (use_grouped_imma && !executes_gateup &&
            !cudaMoEGroupedImma_buildDirectory(
                d_group_counts, d_group_work_directory, num_experts,
                total_slots, group_work_directory_entries, device_idx, stream))
            return false;

        if (use_grouped_imma)
        {
            const auto *down_descriptors = reinterpret_cast<
                const llaminar2::DeviceNativeQuantisedMatrixDesc *>(
                    d_down_desc_table);
            if (!cudaMoEGroupedImma_project(
                    d_scratch_swiglu_int8,
                    d_scratch_swiglu_scales,
                    down_descriptors,
                    d_group_counts,
                    d_group_offsets,
                    d_group_work_directory,
                    projection_weights,
                    d_scratch_down_out,
                    group_work_directory_entries,
                    num_experts,
                    total_slots,
                    down_columns,
                    intermediate,
                    down_codebook_mask,
                    llaminar2::CUDAMoEBatchInvariantPolicy::
                        down_k_partitions,
                    imma_down_columns,
                    device_idx,
                    stream))
            {
                return false;
            }
        }
        else if (use_ordered_down_kpart)
        {
            const int kScatterTileN =
                llaminar2::debugEnv().gemm.cuda_moe_down_ordered_kpart_tile_n;
            constexpr int kReduceTileN = kVerifierDownTileColumns;
            const int kDirectWarpsPerBlock =
                llaminar2::CUDAMoEBatchInvariantPolicy::directDownWarps(top_k);
            const int N = down_columns;
            const int K = intermediate;
            const bool wide_grid_down = N == kWideVerifierDownN && K == kWideVerifierDownK;
            const bool qfn_iq4_down = N == kQFNDecodeGateUpK && K == kQFNDecodeGateUpN;
            const bool interleave_iq4_down = (wide_grid_down || qfn_iq4_down) && is_sm86_cuda_device(device_idx);
            dim3 scatter_block(kScatterTileN);
            dim3 reduce_block(kReduceTileN);
            dim3 direct_block(kDirectWarpsPerBlock * 32);

            for (int token_base = 0; token_base < seq_len; token_base += splitk_tile_rows)
            {
                const int tile_rows = std::min(splitk_tile_rows, seq_len - token_base);
                const int tile_route_slots = tile_rows * top_k;
                const int original_slot_base = token_base * top_k;
                dim3 scatter_grid(
                    (N + kScatterTileN - 1) / kScatterTileN,
                    down_k_partitions,
                    tile_route_slots);
                dim3 route_reduce_grid(
                    (N + kReduceTileN - 1) / kReduceTileN,
                    tile_route_slots);
                dim3 direct_grid((N + 31) / 32, tile_rows);

                // Resolve the immutable reader geometry before recording a
                // node. The narrow and compact variants have separate bodies.
                const auto launch_narrow_down = [&]<uint8_t CodebookId>() {
                    if constexpr (CodebookId == native_formats::Q8_0.codebook_id)
                    {
                        if (N == kShortVerifierDownN && K == kShortVerifierDownK &&
                            is_sm86_cuda_device(device_idx))
                        {
                            grouped_prefill_down_canonical_publish_k32_kernel_short_q8
                                <<<route_reduce_grid, reduce_block, 0, cuda_stream>>>(
                                    d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                                    d_down_desc_table, d_original_to_grouped, d_original_expert_ids,
                                    projection_weights, d_canonical_route_contributions,
                                    original_slot_base, N, K, num_experts);
                            return;
                        }
                    }
                    const auto launch = [&]<bool WideGrid, int FixedReduction = kWideVerifierDownK,
                                            int TileColumns = kReduceTileN>() {
                        dim3 grid = route_reduce_grid;
                        if constexpr (TileColumns != kReduceTileN)
                            grid.x = static_cast<unsigned int>(kWideVerifierDownN / TileColumns);
                        grouped_prefill_down_canonical_publish_k32_kernel<CodebookId, WideGrid, FixedReduction, TileColumns>
                            <<<grid, TileColumns, 0, cuda_stream>>>(
                                d_scratch_swiglu_int8, d_scratch_swiglu_scales,
                                d_down_desc_table, d_original_to_grouped, d_original_expert_ids,
                                projection_weights, d_canonical_route_contributions,
                                original_slot_base, N, K, num_experts);
                    };
                    if constexpr (CodebookId == native_formats::IQ1_S.codebook_id ||
                                  CodebookId == native_formats::IQ1_M.codebook_id ||
                                  CodebookId == native_formats::IQ2_S.codebook_id ||
                                  CodebookId == native_formats::IQ2_XS.codebook_id ||
                                  CodebookId == native_formats::IQ2_XXS.codebook_id ||
                                  CodebookId == native_formats::IQ3_S.codebook_id ||
                                  CodebookId == native_formats::IQ3_XXS.codebook_id)
                    {
                        if (wide_grid_down)
                        {
                            if constexpr (CodebookId == native_formats::IQ1_S.codebook_id ||
                                          CodebookId == native_formats::IQ2_S.codebook_id ||
                                          CodebookId == native_formats::IQ2_XXS.codebook_id ||
                                          CodebookId == native_formats::IQ3_S.codebook_id)
                            {
                                // Complete 256-column tiles amortize route setup and, for IQ3_S,
                                // the cooperative grid load at the measured four-row geometry.
                                if (seq_len == 4 && tile_rows == seq_len && top_k == 8 && is_sm86_cuda_device(device_idx))
                                    return launch.template operator()<true, kWideVerifierDownK,
                                        kWideVerifierDownTileColumns>();
                            }
                            launch.template operator()<true>();
                            return;
                        }
                    }
                    if constexpr (llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::is_iq4_lut)
                    {
                        if (interleave_iq4_down)
                        {
                            if (qfn_iq4_down)
                                return launch.template operator()<true, kQFNDecodeGateUpN>();
                            return launch.template operator()<true>();
                        }
                    }
                    launch.template operator()<false>();
                };

#define LAUNCH_GROUPED_DOWN_ORDERED_KPART(CB)                                      \
    do {                                                                             \
        if (narrow_canonical_down) {                                                  \
            launch_narrow_down.template operator()<CB>();                            \
        } else if (canonical_publication) {                                           \
            grouped_prefill_down_canonical_kpart_scatter_kernel<CB>                  \
                <<<scatter_grid, scatter_block, 0, cuda_stream>>>(                   \
                    d_scratch_swiglu_int8, d_scratch_swiglu_scales,                  \
                    d_down_desc_table, d_original_to_grouped,                        \
                    d_original_expert_ids, d_down_partials,                          \
                    original_slot_base, tile_route_slots, N, K, num_experts,         \
                    down_k_partitions);                                              \
        } else {                                                                     \
            grouped_prefill_down_canonical_kpart_fused_direct_kernel<CB>             \
                <<<direct_grid, direct_block, 0, cuda_stream>>>(                     \
                    d_scratch_swiglu_int8, d_scratch_swiglu_scales,                  \
                    d_down_desc_table, d_original_to_grouped,                        \
                    d_original_expert_ids, d_group_weights, d_output,                \
                    original_slot_base, token_base, tile_rows, top_k, N, K,          \
                    num_experts, down_k_partitions);                                  \
        }                                                                            \
    } while (0)

                bool launched_down = false;
#define LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(CB)                                           \
    do {                                                                                           \
        if (down_codebook_mask & (uint32_t{1} << (CB))) {                                          \
            LAUNCH_GROUPED_DOWN_ORDERED_KPART(CB);                                                  \
            if (!finishGroupedPrefillLaunch("cudaMoE_grouped_down_ordered_kpart_prefill", cuda_stream)) \
                return false;                                                                       \
            launched_down = true;                                                                   \
        }                                                                                           \
    } while (0)

                if (down_codebook_id == kMixedCodebookSentinel)
                {
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART(kMixedCodebookSentinel);
                    if (!finishGroupedPrefillLaunch(
                            "cudaMoE_grouped_down_ordered_kpart_prefill",
                            cuda_stream))
                    {
                        return false;
                    }
                    launched_down = true;
                }
                else
                {
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(0);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(4);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(5);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(6);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(7);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(8);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(9);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(10);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(11);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(12);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(13);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(14);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(15);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(16);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(17);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(19);
                    LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT(
                        llaminar2::kNativeQuantisedExpandedInt8MinCodebook);
                }
                if (!launched_down)
                {
                    std::fprintf(
                        stderr,
                        "[cudaMoE_grouped_prefill_pipeline] unsupported ordered kpart down codebook_id=%u mask=0x%x\n",
                        static_cast<unsigned>(down_codebook_id),
                        static_cast<unsigned>(down_codebook_mask));
                    return false;
                }

#undef LAUNCH_GROUPED_DOWN_ORDERED_KPART_IF_PRESENT
#undef LAUNCH_GROUPED_DOWN_ORDERED_KPART

                if (canonical_publication && !narrow_canonical_down)
                {
                    grouped_prefill_down_canonical_kpart_reduce_kernel<<<
                        route_reduce_grid, reduce_block, 0, cuda_stream>>>(
                        d_down_partials,
                        d_original_to_grouped,
                        projection_weights,
                        d_canonical_route_contributions,
                        original_slot_base,
                        tile_route_slots,
                        N,
                        down_k_partitions);
                    if (!finishGroupedPrefillLaunch(
                            "cudaMoE_grouped_down_ordered_kpart_reduce_prefill",
                            cuda_stream))
                        return false;
                }
            }
        }
        else
        {
            const int requestedTileM =
                llaminar2::debugEnv().gemm.cuda_moe_prefill_tile_m;
            const int kTileM =
                select_grouped_prefill_tile_m(requestedTileM, max_tokens_per_expert);
            const bool tiny_active_verifier =
                active_expert_slots > 0 && max_tokens_per_expert <= 4;
            const int kTileN = tiny_active_verifier ? 64 : 128;
            dim3 grid((down_columns + kTileN - 1) / kTileN,
                      (max_tokens_per_expert + kTileM - 1) / kTileM,
                      expert_grid);
            dim3 block(kTileN);

#define LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, TM)                                                     \
    do {                                                                                           \
        dim3 cb_grid(grid.x, grid.y, expert_grid);                                                  \
        if (kTileN == 64)                                                                            \
            grouped_native_quantised_down_prefill_kernel<CB, TM, 64><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, d_down_desc_table,                  \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_down_out, down_columns, intermediate);                                         \
        else                                                                                         \
            grouped_native_quantised_down_prefill_kernel<CB, TM, 128><<<cb_grid, block, 0, cuda_stream>>>( \
                d_scratch_swiglu_int8, d_scratch_swiglu_scales, d_down_desc_table,                  \
                d_group_counts, d_group_offsets, d_active_expert_ids, active_expert_slots,          \
                d_scratch_down_out, down_columns, intermediate);                                         \
    } while (0)
#define LAUNCH_GROUPED_DOWN_PREFILL(CB)                                                            \
    do {                                                                                           \
        if (kTileM == 16)      LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 16);                              \
        else if (kTileM == 8)  LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 8);                               \
        else if (kTileM == 4)  LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 4);                               \
        else                   LAUNCH_GROUPED_DOWN_PREFILL_TM(CB, 2);                               \
    } while (0)

            bool launched_down = false;
#define LAUNCH_GROUPED_DOWN_IF_PRESENT(CB)                                                         \
    do {                                                                                           \
        if (down_codebook_mask & (uint32_t{1} << (CB))) {                                          \
            LAUNCH_GROUPED_DOWN_PREFILL(CB);                                                        \
            if (!finishGroupedPrefillLaunch("cudaMoE_grouped_down_prefill", cuda_stream))           \
                return false;                                                                       \
            launched_down = true;                                                                   \
        }                                                                                           \
    } while (0)

            LAUNCH_GROUPED_DOWN_IF_PRESENT(0);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(4);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(5);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(6);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(7);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(8);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(9);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(10);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(11);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(12);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(13);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(14);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(15);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(16);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(17);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(19);
            LAUNCH_GROUPED_DOWN_IF_PRESENT(
                llaminar2::kNativeQuantisedExpandedInt8MinCodebook);
            if (!launched_down)
            {
                std::fprintf(stderr,
                             "[cudaMoE_grouped_prefill_pipeline] unsupported down codebook_id=%u mask=0x%x\n",
                             static_cast<unsigned>(down_codebook_id),
                             static_cast<unsigned>(down_codebook_mask));
                return false;
            }

#undef LAUNCH_GROUPED_DOWN_IF_PRESENT
#undef LAUNCH_GROUPED_DOWN_PREFILL
#undef LAUNCH_GROUPED_DOWN_PREFILL_TM
        }

        if (use_grouped_imma)
        {
            constexpr int kTileN = 64;
            dim3 block(kTileN);
            if (canonical_publication)
            {
                const dim3 grid = make_row_major_grid(
                    static_cast<unsigned int>(
                        (down_columns + kTileN - 1) / kTileN),
                    total_slots);
                grouped_prefill_scatter_contributions_canonical_kernel<<<
                    grid,
                    block,
                    0,
                    cuda_stream>>>(
                    d_canonical_route_contributions,
                    d_scratch_down_out,
                    d_original_to_grouped,
                    total_slots,
                    down_columns);
            }
            else
            {
                const dim3 grid = make_row_major_grid(
                    static_cast<unsigned int>(
                        (down_columns + kTileN - 1) / kTileN),
                    seq_len);
                grouped_prefill_scatter_contributions_ordered_kernel<<<
                    grid,
                    block,
                    0,
                    cuda_stream>>>(
                    d_output,
                    d_scratch_down_out,
                    d_original_to_grouped,
                    seq_len,
                    top_k,
                    down_columns);
            }
            if (!finishGroupedPrefillLaunch(
                    "cudaMoE_grouped_imma_scatter_prefill",
                    cuda_stream))
            {
                return false;
            }
        }
        else if (!use_ordered_down_kpart)
        {
            constexpr int kTileN = 64;
            dim3 block(kTileN);
            if (d_original_to_grouped)
            {
                const dim3 grid = make_row_major_grid(
                    static_cast<unsigned int>(
                        (down_columns + kTileN - 1) / kTileN),
                    seq_len);
                grouped_prefill_scatter_weighted_ordered_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_output, d_scratch_down_out,
                    d_original_to_grouped, d_group_weights,
                    seq_len, top_k, down_columns);
            }
            else
            {
                const dim3 grid = make_row_major_grid(
                    static_cast<unsigned int>(
                        (down_columns + kTileN - 1) / kTileN),
                    total_slots);
                grouped_prefill_scatter_weighted_kernel<<<grid, block, 0, cuda_stream>>>(
                    d_output, d_scratch_down_out,
                    d_group_token_indices, d_group_weights,
                    total_slots, seq_len, top_k,
                    grouped_indices_are_route_slots, down_columns);
            }
            if (!finishGroupedPrefillLaunch("cudaMoE_grouped_scatter_prefill", cuda_stream))
                return false;
        }

        return true;
    }

} // extern "C"
