/**
 * @file CUDAMoEGroupedPrefillKernels.cu
 * @brief Device-planned, byte-invariant CUDA IMMA projections for grouped MoE prefill.
 *
 * Long MoE prefill has enough independent grouped rows to make integer tensor
 * cores economical, but route counts are replay-time device state. A compact
 * planner therefore scans the device-owned expert counts and publishes a fixed-
 * capacity directory of populated 16-row tiles. Projection CTAs consume that
 * directory directly and execute exact `mma.sync.m16n8k32` integer products.
 *
 * Integer equality alone is not the MTP contract. Every 32-value block is
 * converted through the shared NativeVNNI FP32 contribution helper, accumulated
 * inside the public serial-M1 K partitions, and then folded in ascending
 * partition order. Consequently row grouping, directory order, and CTA
 * scheduling cannot alter a result byte. There are no atomics, allocations,
 * transfers, default streams, host-authored replay metadata, or synchronization
 * calls in this path.
 */

#include "CUDAMoEGroupedPrefillKernels.h"

#include "CUDANativeVNNIDecodeCommon.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace
{
    constexpr uint32_t kExpertBits = 9u;
    constexpr uint32_t kExpertMask = (uint32_t{1} << kExpertBits) - 1u;
    constexpr uint32_t kUnusedDirectoryEntry = UINT32_MAX;
    constexpr int kRows = llaminar2::cuda::moe::kGroupedImmaTileRows;
    constexpr int kColumnsPerWarp = 8;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kColumnsPerBlock = kColumnsPerWarp * kWarpsPerBlock;
    constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
    constexpr int kMinimumBlocksPerSm = 1024 / kThreadsPerBlock;
    constexpr int kGateUpWarpsPerBlock = 2 * kWarpsPerBlock;
    constexpr int kGateUpThreadsPerBlock = kGateUpWarpsPerBlock * 32;
    constexpr int kGateUpMinimumBlocksPerSm = 4;
    constexpr int kQuantBlock = 32;
    constexpr int kGateUpOperandBytes =
        kRows * kQuantBlock +
        kGateUpWarpsPerBlock * kColumnsPerWarp * kQuantBlock;
    constexpr int kGateUpProjectionBytes =
        2 * kRows * kColumnsPerBlock * static_cast<int>(sizeof(float));
    constexpr int kGateUpSharedBytes =
        kGateUpProjectionBytes > kGateUpOperandBytes
            ? kGateUpProjectionBytes
            : kGateUpOperandBytes;
    constexpr uint32_t kSupportedCodebookMask =
        (uint32_t{1} << 0) |
        (uint32_t{1} << 4) |
        (uint32_t{1} << 5) |
        (uint32_t{1} << 6) |
        (uint32_t{1} << 7) |
        (uint32_t{1} << 8) |
        (uint32_t{1} << 9) |
        (uint32_t{1} << 10) |
        (uint32_t{1} << 11) |
        (uint32_t{1} << 12) |
        (uint32_t{1} << 13) |
        (uint32_t{1} << 14) |
        (uint32_t{1} << 15) |
        (uint32_t{1} << 16) |
        (uint32_t{1} << 17) |
        (uint32_t{1} << 19);

    /** Abort the captured graph when device-owned grouping metadata is invalid. */
    __device__ __forceinline__ void failFastInvalidGroupedImmaState()
    {
        __trap();
    }

    /** Return the semantic directory index encoded across CUDA grid Y/Z. */
    __device__ __forceinline__ int directoryIndexFromGrid()
    {
        return static_cast<int>(
            blockIdx.y + blockIdx.z * static_cast<unsigned int>(gridDim.y));
    }

    /** Return the row owned by one `m16n8k32` accumulator element. */
    __device__ __forceinline__ int mmaFragmentRow(int lane, int element)
    {
        return (element >> 1) * 8 + (lane >> 2);
    }

    /** Return the column owned by one `m16n8k32` accumulator element. */
    __device__ __forceinline__ int mmaFragmentColumn(int lane, int element)
    {
        return (lane & 3) * 2 + (element & 1);
    }

    /**
     * @brief Evaluate the canonical grouped-prefill SiLU expression.
     *
     * This deliberately matches `grouped_prefill_swiglu_quantize_blockwise_kernel`
     * instead of substituting an approximation or algebraically different form.
     * The fused IMMA epilogue must reproduce every serial-row output byte.
     */
    __device__ __forceinline__ float groupedPrefillSilu(float value)
    {
        return value / (1.0f + expf(-value));
    }

    /** Load one row-major 16x32 signed-INT8 A fragment from shared memory. */
    __device__ __forceinline__ void loadMmaA(
        uint32_t fragment[4],
        const int *shared_base,
        int lane)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int kStrideWords = kQuantBlock / sizeof(int32_t);
        const int *address = shared_base +
                             (lane % 16) * kStrideWords +
                             (lane / 16) * 4;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(fragment[0]), "=r"(fragment[1]),
              "=r"(fragment[2]), "=r"(fragment[3])
            : "l"(address));
#else
        (void)fragment;
        (void)shared_base;
        (void)lane;
        failFastInvalidGroupedImmaState();
#endif
    }

    /** Load one column-major 32x8 signed-INT8 B fragment from shared memory. */
    __device__ __forceinline__ void loadMmaB(
        uint32_t fragment[2],
        const int *shared_base,
        int lane)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int kStrideWords = kQuantBlock / sizeof(int32_t);
        const int *address = shared_base +
                             (lane % 8) * kStrideWords +
                             ((lane / 8) * 4) % 8;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(fragment[0]), "=r"(fragment[1])
            : "l"(address));
#else
        (void)fragment;
        (void)shared_base;
        (void)lane;
        failFastInvalidGroupedImmaState();
#endif
    }

    /** Execute one exact signed-INT8 `m16n8k32` integer MMA. */
    __device__ __forceinline__ void mmaM16N8K32(
        int32_t accumulator[4],
        const uint32_t a_fragment[4],
        const uint32_t b_fragment[2])
    {
#if __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%0, %1, %2, %3};\n"
            : "+r"(accumulator[0]), "+r"(accumulator[1]),
              "+r"(accumulator[2]), "+r"(accumulator[3])
            : "r"(a_fragment[0]), "r"(a_fragment[1]),
              "r"(a_fragment[2]), "r"(a_fragment[3]),
              "r"(b_fragment[0]), "r"(b_fragment[1]));
#else
        (void)accumulator;
        (void)a_fragment;
        (void)b_fragment;
        failFastInvalidGroupedImmaState();
#endif
    }

    /**
     * @brief Validate one descriptor against a compile-time projection format.
     *
     * The check is block-uniform because every CTA owns one expert. A malformed
     * descriptor traps the graph instead of silently publishing zeros.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ bool descriptorMatches(
        const llaminar2::DeviceNativeVNNIMatrixDesc &desc,
        int N,
        int K)
    {
        using Traits =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>;
        return desc.payload && desc.scales &&
               desc.n == N && desc.k == K &&
               desc.blocks_per_row == static_cast<uint32_t>(K / kQuantBlock) &&
               desc.codebook_id == CodebookId &&
               (!(Traits::is_asymmetric || Traits::is_dual_scale) || desc.mins) &&
               (!Traits::is_dual_scale_asym || desc.emins);
    }

    /**
     * @brief Publish one MMA dot through the shared byte-exact FP32 contract.
     *
     * MMA changes only how the exact integer terms are obtained. Activation
     * sums, scale/minimum metadata, IQ1 delta terms, and every FP32 rounding
     * boundary remain identical to serial DP4A decode.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float contributionFromMma(
        const int8_t *activation_values,
        int32_t dot_lo,
        int32_t dot_hi,
        const uint8_t *payload,
        const uint16_t *scales,
        const uint16_t *mins,
        const uint32_t *emins,
        size_t linear,
        float activation_scale)
    {
        using Traits =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>;
        const uint16_t secondary_bits =
            mins ? mins[linear] : uint16_t{0};
        const uint32_t emin_bits =
            emins ? emins[linear] : uint32_t{0};
        const uint16_t iq1m_qh = Traits::is_iq1_m
            ? static_cast<uint16_t>(payload[4]) |
                  (static_cast<uint16_t>(payload[5]) << 8)
            : uint16_t{0};
        const int32_t *activation_words =
            reinterpret_cast<const int32_t *>(activation_values);
        const int subgroup_sum0 =
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[0]) +
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[1]);
        const int subgroup_sum1 =
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[2]) +
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[3]);
        const int subgroup_sum2 =
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[4]) +
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[5]);
        const int subgroup_sum3 =
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[6]) +
            llaminar2::cuda_native_vnni::sum_packed_i8(activation_words[7]);
        const int activation_sum_lo = subgroup_sum0 + subgroup_sum1;
        const int activation_sum_hi = subgroup_sum2 + subgroup_sum3;

        return llaminar2::cuda_native_vnni::
            native_vnni_block_contribution_from_reduced_terms_rn<CodebookId>(
                dot_lo,
                dot_hi,
                activation_scale,
                scales[linear],
                secondary_bits,
                emin_bits,
                activation_sum_lo + activation_sum_hi,
                activation_sum_lo,
                activation_sum_hi,
                iq1m_qh,
                subgroup_sum0,
                subgroup_sum1,
                subgroup_sum2,
                subgroup_sum3);
    }

    /**
     * @brief Decode one warp's 32x8 signed-INT8 B fragment into shared memory.
     *
     * IQ2_S and IQ4_NL are the dominant mixed-format expert tensors in the
     * production Qwen 3.6 model. Their payloads naturally split into four
     * independent decode groups per output column, so four adjacent lanes own
     * one column and all 32 lanes perform useful work. This removes the former
     * four-lookups-in-one-lane dependency chain while preserving every decoded
     * byte. Other formats retain their established whole-column decoder until
     * an equally exact cooperative mapping is defined for their payload.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ void decodeMmaWeightFragment(
        const llaminar2::DeviceNativeVNNIMatrixDesc &descriptor,
        int8_t *__restrict__ shared_warp_weights,
        int block,
        int N,
        int column_base,
        int lane)
    {
        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::
                payload_bytes;

        if constexpr (CodebookId == 13)
        {
            const int local_column = lane >> 2;
            const int lookup = lane & 3;
            const int column = column_base + local_column;
            uint64_t signed_grid = 0;
            if (column < N)
            {
                const size_t linear =
                    static_cast<size_t>(block) * N + column;
                const uint8_t *payload =
                    descriptor.payload + linear * kPayloadBytes;
                const uint8_t qh = payload[4];
                const int index =
                    static_cast<int>(payload[lookup]) |
                    (static_cast<int>((qh >> (2 * lookup)) & 0x3u) << 8);
                const uint64_t grid =
                    llaminar2::cuda_native_vnni::iq2_grid_lookup<13>(index);
                const uint8_t signs = payload[5 + lookup];
                const uint32_t low =
                    llaminar2::cuda_native_vnni::iq_apply_signs_4(
                        static_cast<uint32_t>(grid),
                        signs & 0x0Fu);
                const uint32_t high =
                    llaminar2::cuda_native_vnni::iq_apply_signs_4(
                        static_cast<uint32_t>(grid >> 32),
                        signs >> 4);
                signed_grid =
                    static_cast<uint64_t>(low) |
                    (static_cast<uint64_t>(high) << 32);
            }
            *reinterpret_cast<uint64_t *>(
                shared_warp_weights + local_column * kQuantBlock +
                lookup * sizeof(uint64_t)) = signed_grid;
        }
        else if constexpr (CodebookId == 4)
        {
            const int local_column = lane >> 2;
            const int group = lane & 3;
            const int column = column_base + local_column;
            uint32_t low = 0;
            uint32_t high = 0;
            if (column < N)
            {
                const size_t linear =
                    static_cast<size_t>(block) * N + column;
                const uint8_t *payload =
                    descriptor.payload + linear * kPayloadBytes;
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(
                    payload + group * sizeof(uint32_t));
                llaminar2::cuda_native_vnni::iq4nl_decode_word(
                    raw,
                    low,
                    high);
            }
            uint32_t *column_groups = reinterpret_cast<uint32_t *>(
                shared_warp_weights + local_column * kQuantBlock);
            column_groups[group] = low;
            column_groups[group + 4] = high;
        }
        else if (lane < kColumnsPerWarp)
        {
            int32_t packed_groups[8] = {};
            const int column = column_base + lane;
            if (column < N)
            {
                const size_t linear =
                    static_cast<size_t>(block) * N + column;
                llaminar2::cuda_native_vnni::decode_groups_vec<CodebookId>(
                    descriptor.payload + linear * kPayloadBytes,
                    packed_groups);
            }
            *reinterpret_cast<int4 *>(
                shared_warp_weights + lane * kQuantBlock) =
                make_int4(
                    packed_groups[0], packed_groups[1],
                    packed_groups[2], packed_groups[3]);
            *reinterpret_cast<int4 *>(
                shared_warp_weights + lane * kQuantBlock + sizeof(int4)) =
                make_int4(
                    packed_groups[4], packed_groups[5],
                    packed_groups[6], packed_groups[7]);
        }
    }

    /**
     * @brief Build a compact directory of populated 16-row expert tiles.
     *
     * One 512-thread CTA performs a deterministic inclusive scan over expert
     * tile counts. The captured projection grid uses the conservative directory
     * capacity and exits on the explicit sentinel tail, so replay never needs a
     * host-visible live tile count.
     */
    __global__ __launch_bounds__(
        llaminar2::cuda::moe::kGroupedImmaMaximumExperts,
        1) void buildGroupedImmaDirectoryKernel(
        const int *__restrict__ group_counts,
        uint32_t *__restrict__ directory,
        int num_experts,
        int total_slots,
        int directory_entries)
    {
        constexpr int kMaximumExperts =
            llaminar2::cuda::moe::kGroupedImmaMaximumExperts;
        const int tid = static_cast<int>(threadIdx.x);
        if (num_experts <= 0 || num_experts > kMaximumExperts ||
            total_slots <= 0 || directory_entries <= 0)
        {
            if (tid == 0)
                failFastInvalidGroupedImmaState();
            return;
        }

        for (int entry = tid; entry < directory_entries;
             entry += static_cast<int>(blockDim.x))
        {
            directory[entry] = kUnusedDirectoryEntry;
        }

        __shared__ int inclusive_tile_counts[kMaximumExperts];
        const int count = tid < num_experts ? group_counts[tid] : 0;
        if (count < 0 || count > total_slots)
        {
            failFastInvalidGroupedImmaState();
            return;
        }
        inclusive_tile_counts[tid] =
            (count + kRows - 1) / kRows;
        __syncthreads();

        for (int stride = 1; stride < kMaximumExperts; stride <<= 1)
        {
            const int addend = tid >= stride
                                   ? inclusive_tile_counts[tid - stride]
                                   : 0;
            __syncthreads();
            inclusive_tile_counts[tid] += addend;
            __syncthreads();
        }

        const int total_tiles = inclusive_tile_counts[kMaximumExperts - 1];
        if (total_tiles > directory_entries)
        {
            if (tid == 0)
                failFastInvalidGroupedImmaState();
            return;
        }

        if (tid < num_experts && count > 0)
        {
            const int first_tile = tid == 0
                                       ? 0
                                       : inclusive_tile_counts[tid - 1];
            const int expert_tiles = (count + kRows - 1) / kRows;
            for (int tile = 0; tile < expert_tiles; ++tile)
            {
                const uint32_t first_row =
                    static_cast<uint32_t>(tile * kRows);
                directory[first_tile + tile] =
                    (first_row << kExpertBits) |
                    static_cast<uint32_t>(tid);
            }
        }
    }

    /**
     * @brief Project compact expert row tiles with exact INT8 tensor-core MMA.
     *
     * Four independent warps share each CTA's 16 activation rows and own 32
     * consecutive output columns. A weight block is decoded once per output
     * column and reused across all valid rows. K work is processed partition by
     * partition in one CTA; no global partial buffer or cross-CTA reduction is
     * required, and the final FP32 tree is exactly the public serial-M1 tree.
     */
    template <uint8_t CodebookId>
    __global__ __launch_bounds__(kThreadsPerBlock, kMinimumBlocksPerSm)
    void groupedImmaProjectionKernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A,
        const llaminar2::DeviceNativeVNNIMatrixDesc *__restrict__ descriptors,
        const int *__restrict__ group_counts,
        const int *__restrict__ group_offsets,
        const uint32_t *__restrict__ directory,
        const float *__restrict__ partition_weights,
        float *__restrict__ output,
        int directory_entries,
        int num_experts,
        int total_slots,
        int N,
        int K,
        int k_partitions)
    {
        const int directory_index = directoryIndexFromGrid();
        if (directory_index >= directory_entries)
            return;

        const uint32_t packed_tile = directory[directory_index];
        if (packed_tile == kUnusedDirectoryEntry)
            return;

        const int expert = static_cast<int>(packed_tile & kExpertMask);
        const int local_first_row =
            static_cast<int>(packed_tile >> kExpertBits);
        if (expert < 0 || expert >= num_experts)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        const int count = group_counts[expert];
        const int group_offset = group_offsets[expert];
        if (count <= 0 || local_first_row < 0 || local_first_row >= count ||
            group_offset < 0 || group_offset + count > total_slots)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        const auto descriptor = descriptors[expert];
        if (descriptor.codebook_id != CodebookId)
            return;
        if (!descriptorMatches<CodebookId>(descriptor, N, K) ||
            k_partitions <= 0)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        const int active_rows = min(kRows, count - local_first_row);
        const int grouped_row_base = group_offset + local_first_row;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int column_base =
            (static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp) *
            kColumnsPerWarp;

        __shared__ __align__(16) int8_t shared_a[kRows * kQuantBlock];
        __shared__ __align__(16)
            int8_t shared_b[kWarpsPerBlock][kColumnsPerWarp * kQuantBlock];

        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::
                payload_bytes;
        const uint16_t *weight_scales =
            static_cast<const uint16_t *>(descriptor.scales);
        const uint16_t *weight_mins =
            static_cast<const uint16_t *>(descriptor.mins);
        const uint32_t *weight_emins =
            static_cast<const uint32_t *>(descriptor.emins);
        const int blocks_per_row = K / kQuantBlock;
        const int blocks_per_partition =
            (blocks_per_row + k_partitions - 1) / k_partitions;
        float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int partition = 0; partition < k_partitions; ++partition)
        {
            float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int block_begin = partition * blocks_per_partition;
            const int block_end = min(
                blocks_per_row,
                block_begin + blocks_per_partition);

            for (int block = block_begin; block < block_end; ++block)
            {
                /*
                 * Threads 0..31 issue exactly the 32 aligned vector loads that
                 * cover a 16x32 activation tile. Tail rows are explicit zeroes,
                 * so `ldmatrix` always observes a complete physical fragment.
                 */
                if (threadIdx.x < 32)
                {
                    const int local_row = static_cast<int>(threadIdx.x) >> 1;
                    const int half = static_cast<int>(threadIdx.x) & 1;
                    int4 activation = make_int4(0, 0, 0, 0);
                    if (local_row < active_rows)
                    {
                        const int grouped_row = grouped_row_base + local_row;
                        activation = *reinterpret_cast<const int4 *>(
                            A_int8 + static_cast<size_t>(grouped_row) * K +
                            block * kQuantBlock + half * sizeof(int4));
                    }
                    *reinterpret_cast<int4 *>(
                        shared_a + local_row * kQuantBlock +
                        half * sizeof(int4)) = activation;
                }

                decodeMmaWeightFragment<CodebookId>(
                    descriptor,
                    shared_b[warp],
                    block,
                    N,
                    column_base,
                    lane);
                __syncthreads();

                uint32_t a_fragment[4];
                uint32_t b_fragment[2];
                loadMmaA(
                    a_fragment,
                    reinterpret_cast<const int *>(shared_a),
                    lane);
                loadMmaB(
                    b_fragment,
                    reinterpret_cast<const int *>(shared_b[warp]),
                    lane);

                int32_t dot_lo[4] = {0, 0, 0, 0};
                int32_t dot_hi[4] = {0, 0, 0, 0};
                if constexpr (llaminar2::cuda_native_vnni::
                                  CodebookTraits<CodebookId>::is_dual_scale)
                {
                    const uint32_t b_lo[2] = {b_fragment[0], 0u};
                    const uint32_t b_hi[2] = {0u, b_fragment[1]};
                    mmaM16N8K32(dot_lo, a_fragment, b_lo);
                    mmaM16N8K32(dot_hi, a_fragment, b_hi);
                }
                else
                {
                    mmaM16N8K32(dot_lo, a_fragment, b_fragment);
                }

#pragma unroll
                for (int element = 0; element < 4; ++element)
                {
                    const int local_row = mmaFragmentRow(lane, element);
                    const int column =
                        column_base + mmaFragmentColumn(lane, element);
                    if (local_row >= active_rows || column >= N)
                        continue;

                    const int grouped_row = grouped_row_base + local_row;
                    const size_t linear =
                        static_cast<size_t>(block) * N + column;
                    const uint8_t *payload =
                        descriptor.payload + linear * kPayloadBytes;
                    const float activation_scale = scales_A[
                        static_cast<size_t>(grouped_row) * blocks_per_row +
                        block];
                    const float contribution = contributionFromMma<CodebookId>(
                        shared_a + local_row * kQuantBlock,
                        dot_lo[element],
                        dot_hi[element],
                        payload,
                        weight_scales,
                        weight_mins,
                        weight_emins,
                        linear,
                        activation_scale);
                    partial[element] = __fadd_rn(
                        partial[element],
                        contribution);
                }
                __syncthreads();
            }

#pragma unroll
            for (int element = 0; element < 4; ++element)
            {
                const int local_row = mmaFragmentRow(lane, element);
                float partition_contribution = partial[element];
                if (partition_weights && local_row < active_rows)
                {
                    partition_contribution = __fmul_rn(
                        partition_weights[grouped_row_base + local_row],
                        partition_contribution);
                }
                total[element] = __fadd_rn(
                    total[element],
                    partition_contribution);
            }
        }

#pragma unroll
        for (int element = 0; element < 4; ++element)
        {
            const int local_row = mmaFragmentRow(lane, element);
            const int column =
                column_base + mmaFragmentColumn(lane, element);
            if (local_row < active_rows && column < N)
            {
                output[
                    static_cast<size_t>(grouped_row_base + local_row) * N +
                    column] = total[element];
            }
        }
    }

    /**
     * @brief Fuse grouped gate/up IMMA and exact SwiGLU requantization.
     *
     * One CTA owns the same `(expert, 16 rows, 32 columns)` tile for both
     * projections. Warps 0--3 execute gate while warps 4--7 execute up, so the
     * two projections retain the proven four-warp MMA geometry and register
     * footprint. The CTA loads each activation fragment once for all eight
     * warps. After the canonical K-partition folds complete, the operand arena
     * is reused for FP32 gate/up fragments and each warp quantizes two rows with
     * the exact standalone SwiGLU reduction order.
     *
     * This is the sole production gate/up path for grouped IMMA. It removes the
     * second activation traversal, both FP32 global intermediates, and the
     * standalone SwiGLU launch. No arithmetic operation is fused across a
     * serial-decode rounding boundary: gate and up retain independent integer
     * MMA and FP32 partition trees before the canonical nonlinear epilogue.
     */
    template <uint8_t CodebookId>
    __global__ __launch_bounds__(
        kGateUpThreadsPerBlock,
        kGateUpMinimumBlocksPerSm)
    void groupedImmaGateUpSwiGluKernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A,
        const llaminar2::DeviceNativeVNNIMatrixDesc *__restrict__ gate_descriptors,
        const llaminar2::DeviceNativeVNNIMatrixDesc *__restrict__ up_descriptors,
        const int *__restrict__ group_counts,
        const int *__restrict__ group_offsets,
        const uint32_t *__restrict__ directory,
        int8_t *__restrict__ swiglu_int8,
        float *__restrict__ swiglu_scales,
        int directory_entries,
        int num_experts,
        int total_slots,
        int N,
        int K,
        int k_partitions)
    {
        const int directory_index = directoryIndexFromGrid();
        if (directory_index >= directory_entries)
            return;

        const uint32_t packed_tile = directory[directory_index];
        if (packed_tile == kUnusedDirectoryEntry)
            return;

        const int expert = static_cast<int>(packed_tile & kExpertMask);
        const int local_first_row =
            static_cast<int>(packed_tile >> kExpertBits);
        if (expert < 0 || expert >= num_experts)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        const int count = group_counts[expert];
        const int group_offset = group_offsets[expert];
        if (count <= 0 || local_first_row < 0 || local_first_row >= count ||
            group_offset < 0 || group_offset + count > total_slots)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        /*
         * Gate/up table upload rejects unlike codebooks for one expert. Testing
         * the gate id therefore gives every thread the same specialization
         * decision and makes the early exit safe before any CTA barrier.
         */
        if (gate_descriptors[expert].codebook_id != CodebookId)
            return;

        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const bool is_up_projection = warp >= kWarpsPerBlock;
        const int projection_warp = warp & (kWarpsPerBlock - 1);
        const auto descriptor = is_up_projection
                                    ? up_descriptors[expert]
                                    : gate_descriptors[expert];
        if (!descriptorMatches<CodebookId>(descriptor, N, K) ||
            descriptor.codebook_id != CodebookId || k_partitions <= 0)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        const int active_rows = min(kRows, count - local_first_row);
        const int grouped_row_base = group_offset + local_first_row;
        const int column_base =
            (static_cast<int>(blockIdx.x) * kWarpsPerBlock +
             projection_warp) *
            kColumnsPerWarp;

        static_assert((kGateUpSharedBytes % sizeof(uint32_t)) == 0);
        __shared__ __align__(16) uint32_t shared_storage[
            kGateUpSharedBytes / sizeof(uint32_t)];
        auto *shared_bytes = reinterpret_cast<int8_t *>(shared_storage);
        int8_t *shared_a = shared_bytes;
        int8_t *shared_b =
            shared_bytes + kRows * kQuantBlock +
            warp * kColumnsPerWarp * kQuantBlock;

        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::
                payload_bytes;
        const uint16_t *weight_scales =
            static_cast<const uint16_t *>(descriptor.scales);
        const uint16_t *weight_mins =
            static_cast<const uint16_t *>(descriptor.mins);
        const uint32_t *weight_emins =
            static_cast<const uint32_t *>(descriptor.emins);
        const int blocks_per_row = K / kQuantBlock;
        const int blocks_per_partition =
            (blocks_per_row + k_partitions - 1) / k_partitions;
        float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int partition = 0; partition < k_partitions; ++partition)
        {
            float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int block_begin = partition * blocks_per_partition;
            const int block_end = min(
                blocks_per_row,
                block_begin + blocks_per_partition);

            for (int block = block_begin; block < block_end; ++block)
            {
                if (threadIdx.x < 32)
                {
                    const int local_row = static_cast<int>(threadIdx.x) >> 1;
                    const int half = static_cast<int>(threadIdx.x) & 1;
                    int4 activation = make_int4(0, 0, 0, 0);
                    if (local_row < active_rows)
                    {
                        const int grouped_row = grouped_row_base + local_row;
                        activation = *reinterpret_cast<const int4 *>(
                            A_int8 + static_cast<size_t>(grouped_row) * K +
                            block * kQuantBlock + half * sizeof(int4));
                    }
                    *reinterpret_cast<int4 *>(
                        shared_a + local_row * kQuantBlock +
                        half * sizeof(int4)) = activation;
                }

                decodeMmaWeightFragment<CodebookId>(
                    descriptor,
                    shared_b,
                    block,
                    N,
                    column_base,
                    lane);
                __syncthreads();

                uint32_t a_fragment[4];
                uint32_t b_fragment[2];
                loadMmaA(
                    a_fragment,
                    reinterpret_cast<const int *>(shared_a),
                    lane);
                loadMmaB(
                    b_fragment,
                    reinterpret_cast<const int *>(shared_b),
                    lane);

                int32_t dot_lo[4] = {0, 0, 0, 0};
                int32_t dot_hi[4] = {0, 0, 0, 0};
                if constexpr (llaminar2::cuda_native_vnni::
                                  CodebookTraits<CodebookId>::is_dual_scale)
                {
                    const uint32_t b_lo[2] = {b_fragment[0], 0u};
                    const uint32_t b_hi[2] = {0u, b_fragment[1]};
                    mmaM16N8K32(dot_lo, a_fragment, b_lo);
                    mmaM16N8K32(dot_hi, a_fragment, b_hi);
                }
                else
                {
                    mmaM16N8K32(dot_lo, a_fragment, b_fragment);
                }

#pragma unroll
                for (int element = 0; element < 4; ++element)
                {
                    const int local_row = mmaFragmentRow(lane, element);
                    const int column =
                        column_base + mmaFragmentColumn(lane, element);
                    if (local_row >= active_rows || column >= N)
                        continue;

                    const int grouped_row = grouped_row_base + local_row;
                    const size_t linear =
                        static_cast<size_t>(block) * N + column;
                    const uint8_t *payload =
                        descriptor.payload + linear * kPayloadBytes;
                    const float activation_scale = scales_A[
                        static_cast<size_t>(grouped_row) * blocks_per_row +
                        block];
                    const float contribution = contributionFromMma<CodebookId>(
                        shared_a + local_row * kQuantBlock,
                        dot_lo[element],
                        dot_hi[element],
                        payload,
                        weight_scales,
                        weight_mins,
                        weight_emins,
                        linear,
                        activation_scale);
                    partial[element] = __fadd_rn(
                        partial[element],
                        contribution);
                }
                __syncthreads();
            }

#pragma unroll
            for (int element = 0; element < 4; ++element)
            {
                total[element] = __fadd_rn(
                    total[element],
                    partial[element]);
            }
        }

        /*
         * The final loop barrier makes the operand lifetime end uniformly.
         * Reuse the same shared allocation for gate/up FP32 fragments so this
         * fusion does not trade global traffic for a larger residency burden.
         */
        float *shared_projections =
            reinterpret_cast<float *>(shared_storage);
        const int projection = is_up_projection ? 1 : 0;
#pragma unroll
        for (int element = 0; element < 4; ++element)
        {
            const int local_row = mmaFragmentRow(lane, element);
            const int local_column =
                projection_warp * kColumnsPerWarp +
                mmaFragmentColumn(lane, element);
            const int column =
                static_cast<int>(blockIdx.x) * kColumnsPerBlock + local_column;
            if (local_row < active_rows && column < N)
            {
                shared_projections[
                    (projection * kRows + local_row) * kColumnsPerBlock +
                    local_column] = total[element];
            }
        }
        __syncthreads();

        const int quant_column =
            static_cast<int>(blockIdx.x) * kColumnsPerBlock + lane;
        const bool quant_column_active = quant_column < N;
        const int output_blocks_per_row =
            (N + kColumnsPerBlock - 1) / kColumnsPerBlock;
        for (int local_row = warp; local_row < active_rows;
             local_row += kGateUpWarpsPerBlock)
        {
            const int fragment_index =
                local_row * kColumnsPerBlock + lane;
            const float gate = quant_column_active
                                   ? shared_projections[fragment_index]
                                   : 0.0f;
            const float up = quant_column_active
                                 ? shared_projections[
                                       kRows * kColumnsPerBlock + fragment_index]
                                 : 0.0f;
            const float value = quant_column_active
                                    ? groupedPrefillSilu(gate) * up
                                    : 0.0f;

            float abs_value = fabsf(value);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
            {
                abs_value = fmaxf(
                    abs_value,
                    __shfl_xor_sync(0xffffffffu, abs_value, mask));
            }

            const float scale =
                abs_value > 0.0f ? abs_value / 127.0f : 1.0f;
            if (quant_column_active)
            {
                const int grouped_row = grouped_row_base + local_row;
                if (lane == 0)
                {
                    swiglu_scales[
                        static_cast<size_t>(grouped_row) *
                            output_blocks_per_row +
                        static_cast<int>(blockIdx.x)] = scale;
                }
                const float quantized = value / scale;
                swiglu_int8[
                    static_cast<size_t>(grouped_row) * N + quant_column] =
                    static_cast<int8_t>(rintf(fminf(
                        127.0f,
                        fmaxf(-127.0f, quantized))));
            }
        }
    }

    /** Build a legal CUDA grid for an arbitrarily long compact directory. */
    bool makeProjectionGrid(
        int N,
        int directory_entries,
        dim3 &grid)
    {
        constexpr unsigned int kGridDimensionLimit = 65535u;
        if (N <= 0 || directory_entries <= 0)
            return false;
        const unsigned int rows =
            static_cast<unsigned int>(directory_entries);
        const unsigned int rows_y = std::min(rows, kGridDimensionLimit);
        const unsigned int rows_z = (rows + rows_y - 1u) / rows_y;
        if (rows_z == 0 || rows_z > kGridDimensionLimit)
            return false;
        grid = dim3(
            static_cast<unsigned int>((N + kColumnsPerBlock - 1) /
                                      kColumnsPerBlock),
            rows_y,
            rows_z);
        return grid.x > 0;
    }

    /** Launch one codebook specialization when its table mask bit is present. */
    template <uint8_t CodebookId>
    bool launchCodebookProjection(
        uint32_t codebook_mask,
        const dim3 &grid,
        cudaStream_t stream,
        const int8_t *A_int8,
        const float *scales_A,
        const llaminar2::DeviceNativeVNNIMatrixDesc *descriptors,
        const int *group_counts,
        const int *group_offsets,
        const uint32_t *directory,
        const float *partition_weights,
        float *output,
        int directory_entries,
        int num_experts,
        int total_slots,
        int N,
        int K,
        int k_partitions,
        bool &launched)
    {
        if ((codebook_mask & (uint32_t{1} << CodebookId)) == 0)
            return true;
        groupedImmaProjectionKernel<CodebookId><<<
            grid,
            kThreadsPerBlock,
            0,
            stream>>>(
            A_int8,
            scales_A,
            descriptors,
            group_counts,
            group_offsets,
            directory,
            partition_weights,
            output,
            directory_entries,
            num_experts,
            total_slots,
            N,
            K,
            k_partitions);
        launched = true;
        return cudaGetLastError() == cudaSuccess;
    }

    /** Launch one fused gate/up/SwiGLU specialization when its mask bit exists. */
    template <uint8_t CodebookId>
    bool launchCodebookGateUpSwiGlu(
        uint32_t codebook_mask,
        const dim3 &grid,
        cudaStream_t stream,
        const int8_t *A_int8,
        const float *scales_A,
        const llaminar2::DeviceNativeVNNIMatrixDesc *gate_descriptors,
        const llaminar2::DeviceNativeVNNIMatrixDesc *up_descriptors,
        const int *group_counts,
        const int *group_offsets,
        const uint32_t *directory,
        int8_t *swiglu_int8,
        float *swiglu_scales,
        int directory_entries,
        int num_experts,
        int total_slots,
        int N,
        int K,
        int k_partitions,
        bool &launched)
    {
        if ((codebook_mask & (uint32_t{1} << CodebookId)) == 0)
            return true;
        groupedImmaGateUpSwiGluKernel<CodebookId><<<
            grid,
            kGateUpThreadsPerBlock,
            0,
            stream>>>(
            A_int8,
            scales_A,
            gate_descriptors,
            up_descriptors,
            group_counts,
            group_offsets,
            directory,
            swiglu_int8,
            swiglu_scales,
            directory_entries,
            num_experts,
            total_slots,
            N,
            K,
            k_partitions);
        launched = true;
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace

extern "C" bool cudaMoEGroupedImma_buildDirectory(
    const int *d_group_counts,
    uint32_t *d_directory,
    int num_experts,
    int total_slots,
    int directory_entries,
    int device_idx,
    void *stream)
{
    if (!d_group_counts || !d_directory || !stream ||
        num_experts <= 0 ||
        num_experts > llaminar2::cuda::moe::kGroupedImmaMaximumExperts ||
        total_slots <= 0 || directory_entries <= 0 ||
        static_cast<std::size_t>(directory_entries) <
            llaminar2::cuda::moe::groupedImmaDirectoryEntries(
                static_cast<std::size_t>(total_slots),
                static_cast<std::size_t>(num_experts)))
    {
        return false;
    }
    if (cudaSetDevice(device_idx) != cudaSuccess)
        return false;

    buildGroupedImmaDirectoryKernel<<<
        1,
        llaminar2::cuda::moe::kGroupedImmaMaximumExperts,
        0,
        static_cast<cudaStream_t>(stream)>>>(
        d_group_counts,
        d_directory,
        num_experts,
        total_slots,
        directory_entries);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool cudaMoEGroupedImma_project(
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
    void *stream)
{
    if (!d_A_int8 || !d_scales_A || !d_desc_table ||
        !d_group_counts || !d_group_offsets || !d_directory || !d_output ||
        !stream || directory_entries <= 0 || num_experts <= 0 ||
        num_experts > llaminar2::cuda::moe::kGroupedImmaMaximumExperts ||
        total_slots <= 0 || N <= 0 || K <= 0 || (K % kQuantBlock) != 0 ||
        codebook_mask == 0 || (codebook_mask & ~kSupportedCodebookMask) != 0 ||
        k_partitions <= 0)
    {
        return false;
    }
    if (cudaSetDevice(device_idx) != cudaSuccess)
        return false;

    dim3 grid;
    if (!makeProjectionGrid(N, directory_entries, grid))
        return false;

    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    bool launched = false;
#define LAUNCH_CODEBOOK(CB)                                                   \
    do                                                                         \
    {                                                                          \
        if (!launchCodebookProjection<CB>(                                    \
                codebook_mask, grid, cuda_stream, d_A_int8, d_scales_A,       \
                d_desc_table, d_group_counts, d_group_offsets, d_directory,   \
                d_partition_weights, d_output, directory_entries,             \
                num_experts, total_slots, N, K, k_partitions, launched))      \
        {                                                                      \
            return false;                                                      \
        }                                                                      \
    } while (0)

    LAUNCH_CODEBOOK(0);
    LAUNCH_CODEBOOK(4);
    LAUNCH_CODEBOOK(5);
    LAUNCH_CODEBOOK(6);
    LAUNCH_CODEBOOK(7);
    LAUNCH_CODEBOOK(8);
    LAUNCH_CODEBOOK(9);
    LAUNCH_CODEBOOK(10);
    LAUNCH_CODEBOOK(11);
    LAUNCH_CODEBOOK(12);
    LAUNCH_CODEBOOK(13);
    LAUNCH_CODEBOOK(14);
    LAUNCH_CODEBOOK(15);
    LAUNCH_CODEBOOK(16);
    LAUNCH_CODEBOOK(17);
    LAUNCH_CODEBOOK(19);

#undef LAUNCH_CODEBOOK
    return launched;
}

extern "C" bool cudaMoEGroupedImma_projectGateUpSwiGlu(
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
    void *stream)
{
    if (!d_A_int8 || !d_scales_A || !d_gate_desc_table ||
        !d_up_desc_table || !d_group_counts || !d_group_offsets ||
        !d_directory || !d_swiglu_int8 || !d_swiglu_scales || !stream ||
        directory_entries <= 0 || num_experts <= 0 ||
        num_experts > llaminar2::cuda::moe::kGroupedImmaMaximumExperts ||
        total_slots <= 0 || N <= 0 || K <= 0 ||
        (N % kColumnsPerBlock) != 0 || (K % kQuantBlock) != 0 ||
        codebook_mask == 0 ||
        (codebook_mask & ~kSupportedCodebookMask) != 0 ||
        k_partitions <= 0)
    {
        return false;
    }
    if (cudaSetDevice(device_idx) != cudaSuccess)
        return false;

    dim3 grid;
    if (!makeProjectionGrid(N, directory_entries, grid))
        return false;

    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    bool launched = false;
#define LAUNCH_GATE_UP_CODEBOOK(CB)                                            \
    do                                                                          \
    {                                                                           \
        if (!launchCodebookGateUpSwiGlu<CB>(                                   \
                codebook_mask, grid, cuda_stream, d_A_int8, d_scales_A,        \
                d_gate_desc_table, d_up_desc_table, d_group_counts,            \
                d_group_offsets, d_directory, d_swiglu_int8,                   \
                d_swiglu_scales, directory_entries, num_experts, total_slots,  \
                N, K, k_partitions, launched))                                 \
        {                                                                       \
            return false;                                                       \
        }                                                                       \
    } while (0)

    LAUNCH_GATE_UP_CODEBOOK(0);
    LAUNCH_GATE_UP_CODEBOOK(4);
    LAUNCH_GATE_UP_CODEBOOK(5);
    LAUNCH_GATE_UP_CODEBOOK(6);
    LAUNCH_GATE_UP_CODEBOOK(7);
    LAUNCH_GATE_UP_CODEBOOK(8);
    LAUNCH_GATE_UP_CODEBOOK(9);
    LAUNCH_GATE_UP_CODEBOOK(10);
    LAUNCH_GATE_UP_CODEBOOK(11);
    LAUNCH_GATE_UP_CODEBOOK(12);
    LAUNCH_GATE_UP_CODEBOOK(13);
    LAUNCH_GATE_UP_CODEBOOK(14);
    LAUNCH_GATE_UP_CODEBOOK(15);
    LAUNCH_GATE_UP_CODEBOOK(16);
    LAUNCH_GATE_UP_CODEBOOK(17);
    LAUNCH_GATE_UP_CODEBOOK(19);

#undef LAUNCH_GATE_UP_CODEBOOK
    return launched;
}

namespace
{
    /** Query compiler resources for one exact IMMA specialization. */
    template <uint8_t CodebookId>
    bool queryGroupedImmaKernelResources(
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(
                &attributes,
                groupedImmaProjectionKernel<CodebookId>) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                groupedImmaProjectionKernel<CodebookId>,
                kThreadsPerBlock,
                0) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        *registers_per_thread = attributes.numRegs;
        *local_memory_bytes_per_thread = attributes.localSizeBytes;
        *static_shared_memory_bytes = attributes.sharedSizeBytes;
        *max_threads_per_block = attributes.maxThreadsPerBlock;
        *max_active_blocks_per_sm = active_blocks;
        return active_blocks > 0;
    }

    /** Query compiler resources for one fused gate/up/SwiGLU specialization. */
    template <uint8_t CodebookId>
    bool queryGroupedImmaGateUpKernelResources(
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(
                &attributes,
                groupedImmaGateUpSwiGluKernel<CodebookId>) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                groupedImmaGateUpSwiGluKernel<CodebookId>,
                kGateUpThreadsPerBlock,
                0) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        *registers_per_thread = attributes.numRegs;
        *local_memory_bytes_per_thread = attributes.localSizeBytes;
        *static_shared_memory_bytes = attributes.sharedSizeBytes;
        *max_threads_per_block = attributes.maxThreadsPerBlock;
        *max_active_blocks_per_sm = active_blocks;
        return active_blocks > 0;
    }
}

extern "C" bool cudaMoEGroupedImma_queryKernelResources(
    uint8_t codebook_id,
    int *registers_per_thread,
    std::size_t *local_memory_bytes_per_thread,
    std::size_t *static_shared_memory_bytes,
    int *max_threads_per_block,
    int *max_active_blocks_per_sm)
{
    if (!registers_per_thread || !local_memory_bytes_per_thread ||
        !static_shared_memory_bytes || !max_threads_per_block ||
        !max_active_blocks_per_sm)
    {
        return false;
    }

#define QUERY_CODEBOOK(CB)                                                    \
    case CB:                                                                  \
        return queryGroupedImmaKernelResources<CB>(                           \
            registers_per_thread,                                             \
            local_memory_bytes_per_thread,                                    \
            static_shared_memory_bytes,                                       \
            max_threads_per_block,                                            \
            max_active_blocks_per_sm)

    switch (codebook_id)
    {
        QUERY_CODEBOOK(0);
        QUERY_CODEBOOK(4);
        QUERY_CODEBOOK(5);
        QUERY_CODEBOOK(6);
        QUERY_CODEBOOK(7);
        QUERY_CODEBOOK(8);
        QUERY_CODEBOOK(9);
        QUERY_CODEBOOK(10);
        QUERY_CODEBOOK(11);
        QUERY_CODEBOOK(12);
        QUERY_CODEBOOK(13);
        QUERY_CODEBOOK(14);
        QUERY_CODEBOOK(15);
        QUERY_CODEBOOK(16);
        QUERY_CODEBOOK(17);
        QUERY_CODEBOOK(19);
    default:
        return false;
    }

#undef QUERY_CODEBOOK
}

extern "C" bool cudaMoEGroupedImma_queryGateUpKernelResources(
    uint8_t codebook_id,
    int *registers_per_thread,
    std::size_t *local_memory_bytes_per_thread,
    std::size_t *static_shared_memory_bytes,
    int *max_threads_per_block,
    int *max_active_blocks_per_sm)
{
    if (!registers_per_thread || !local_memory_bytes_per_thread ||
        !static_shared_memory_bytes || !max_threads_per_block ||
        !max_active_blocks_per_sm)
    {
        return false;
    }

#define QUERY_GATE_UP_CODEBOOK(CB)                                            \
    case CB:                                                                  \
        return queryGroupedImmaGateUpKernelResources<CB>(                     \
            registers_per_thread,                                             \
            local_memory_bytes_per_thread,                                    \
            static_shared_memory_bytes,                                       \
            max_threads_per_block,                                            \
            max_active_blocks_per_sm)

    switch (codebook_id)
    {
        QUERY_GATE_UP_CODEBOOK(0);
        QUERY_GATE_UP_CODEBOOK(4);
        QUERY_GATE_UP_CODEBOOK(5);
        QUERY_GATE_UP_CODEBOOK(6);
        QUERY_GATE_UP_CODEBOOK(7);
        QUERY_GATE_UP_CODEBOOK(8);
        QUERY_GATE_UP_CODEBOOK(9);
        QUERY_GATE_UP_CODEBOOK(10);
        QUERY_GATE_UP_CODEBOOK(11);
        QUERY_GATE_UP_CODEBOOK(12);
        QUERY_GATE_UP_CODEBOOK(13);
        QUERY_GATE_UP_CODEBOOK(14);
        QUERY_GATE_UP_CODEBOOK(15);
        QUERY_GATE_UP_CODEBOOK(16);
        QUERY_GATE_UP_CODEBOOK(17);
        QUERY_GATE_UP_CODEBOOK(19);
    default:
        return false;
    }

#undef QUERY_GATE_UP_CODEBOOK
}
