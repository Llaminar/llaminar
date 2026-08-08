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
    constexpr int kQuantBlock = 32;

    /** Compile-time resources derived from one arithmetic-neutral CTA width. */
    template <int ColumnsPerBlock>
    struct GroupedImmaGeometry final
    {
        static_assert(ColumnsPerBlock >= 32);
        static_assert((ColumnsPerBlock % kColumnsPerWarp) == 0);
        static_assert((ColumnsPerBlock % kQuantBlock) == 0);

        static constexpr int projection_warps =
            ColumnsPerBlock / kColumnsPerWarp;
        static constexpr int projection_threads = projection_warps * 32;
        static constexpr int projection_minimum_blocks_per_sm =
            1024 / projection_threads > 0
                ? 1024 / projection_threads
                : 1;
    };

    /** Resources for the disjoint gate/up warp-bank schedule. */
    template <int ColumnsPerBlock>
    struct ParallelGateUpGeometry final
    {
        using Projection = GroupedImmaGeometry<ColumnsPerBlock>;
        static constexpr int gate_up_warps = 2 * Projection::projection_warps;
        static constexpr int gate_up_threads = gate_up_warps * 32;
        static constexpr int gate_up_minimum_blocks_per_sm =
            1024 / gate_up_threads > 0
                ? 1024 / gate_up_threads
                : 1;
        static constexpr int gate_up_operand_bytes =
            kRows * kQuantBlock +
            gate_up_warps * kColumnsPerWarp * kQuantBlock;
        static constexpr int gate_up_projection_bytes =
            2 * kRows * ColumnsPerBlock * static_cast<int>(sizeof(float));
        static constexpr int gate_up_shared_bytes =
            gate_up_projection_bytes > gate_up_operand_bytes
                ? gate_up_projection_bytes
                : gate_up_operand_bytes;
    };

    /** Resources for the register-resident paired-projection schedule. */
    template <int ColumnsPerBlock>
    struct PairedGateUpGeometry final
    {
        using Projection = GroupedImmaGeometry<ColumnsPerBlock>;
        static constexpr int gate_up_warps = Projection::projection_warps;
        static constexpr int gate_up_threads = Projection::projection_threads;
        static constexpr int gate_up_minimum_blocks_per_sm =
            Projection::projection_minimum_blocks_per_sm;
        static constexpr int gate_up_operand_bytes =
            kRows * kQuantBlock +
            2 * gate_up_warps * kColumnsPerWarp * kQuantBlock;
        static constexpr int gate_up_value_bytes =
            kRows * ColumnsPerBlock * static_cast<int>(sizeof(float));
        static constexpr int gate_up_shared_bytes =
            gate_up_value_bytes > gate_up_operand_bytes
                ? gate_up_value_bytes
                : gate_up_operand_bytes;
    };

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
     * @brief Publish an ordinary dual-scale MMA dot from staged raw FP16 bits.
     *
     * IQ2_S uses independent scales for the low and high sixteen-value halves
     * but has no minimum or IQ1 correction term. The paired gate/up kernel can
     * therefore stage the two raw scale words once per output column and reuse
     * them for all sixteen rows. This helper deliberately enters the same
     * canonical contribution function as the generic path; shared-memory
     * placement changes where metadata is loaded, never its arithmetic.
     *
     * @tparam CodebookId NativeVNNI ordinary dual-scale codebook identifier.
     * @param dot_lo Exact low-half integer dot product from IMMA.
     * @param dot_hi Exact high-half integer dot product from IMMA.
     * @param activation_scale Exact FP32 activation scale for this row/block.
     * @param packed_scales Low-scale bits in bits 0--15 and high-scale bits in
     *        bits 16--31.
     * @return Canonically rounded FP32 contribution for this quant block.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float contributionFromStagedDualScaleMma(
        int32_t dot_lo,
        int32_t dot_hi,
        float activation_scale,
        uint32_t packed_scales)
    {
        using Traits =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>;
        static_assert(Traits::is_dual_scale);
        static_assert(!Traits::is_dual_scale_asym);
        static_assert(!Traits::is_iq1_m);
        return llaminar2::cuda_native_vnni::
            native_vnni_block_contribution_from_reduced_terms_rn<CodebookId>(
                dot_lo,
                dot_hi,
                activation_scale,
                static_cast<uint16_t>(packed_scales),
                static_cast<uint16_t>(packed_scales >> 16),
                /*emin_bits=*/0,
                /*activation_sum=*/0,
                /*activation_sum_lo=*/0,
                /*activation_sum_hi=*/0,
                /*iq1m_qh=*/0,
                /*subgroup_sum0=*/0,
                /*subgroup_sum1=*/0,
                /*subgroup_sum2=*/0,
                /*subgroup_sum3=*/0);
    }

    /**
     * @brief Publish a symmetric MMA dot from one staged raw FP16 scale.
     *
     * Symmetric codebooks have no correction metadata, so a scale loaded once
     * per output column can be shared by all sixteen row owners. The call still
     * enters the canonical NativeVNNI FP32 publication contract and therefore
     * preserves every serial-row rounding boundary.
     *
     * @tparam CodebookId NativeVNNI symmetric codebook identifier.
     * @param dot Exact integer dot product from IMMA.
     * @param activation_scale Exact FP32 activation scale for this row/block.
     * @param scale_bits Raw FP16 weight-scale bits.
     * @return Canonically rounded FP32 contribution for this quant block.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float contributionFromStagedSymmetricMma(
        int32_t dot,
        float activation_scale,
        uint16_t scale_bits)
    {
        using Traits =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>;
        static_assert(!Traits::is_asymmetric);
        static_assert(!Traits::is_dual_scale);
        static_assert(!Traits::is_iq1_m);
        return llaminar2::cuda_native_vnni::
            native_vnni_block_contribution_from_reduced_terms_rn<CodebookId>(
                dot,
                /*dot_hi=*/0,
                activation_scale,
                scale_bits,
                /*secondary_bits=*/0,
                /*emin_bits=*/0,
                /*activation_sum=*/0,
                /*activation_sum_lo=*/0,
                /*activation_sum_hi=*/0,
                /*iq1m_qh=*/0,
                /*subgroup_sum0=*/0,
                /*subgroup_sum1=*/0,
                /*subgroup_sum2=*/0,
                /*subgroup_sum3=*/0);
    }

    /**
     * @brief Decode paired IQ2_S gate/up fragments with independent loads in flight.
     *
     * Gate and up payloads use unrelated immutable weight arrays, so their
     * codebook lookups have no ordering dependency. Issuing both random
     * read-only-table loads before applying either sign mask gives the SM two
     * independent misses to overlap. The decoded bytes and their shared-memory
     * destinations are exactly the same as two calls to the ordinary IQ2_S
     * fragment decoder; only instruction scheduling changes.
     *
     * @param gate_payload_base Device-owned gate payload allocation.
     * @param up_payload_base Device-owned up payload allocation.
     * @param shared_gate_weights Warp-private gate destination.
     * @param shared_up_weights Warp-private up destination.
     * @param block Quantization block along K.
     * @param N Projection output width.
     * @param column_base First output column owned by this warp.
     * @param lane Lane index within the warp.
     */
    __device__ __forceinline__ void decodePairedIQ2SWeightFragments(
        const uint8_t *__restrict__ gate_payload_base,
        const uint8_t *__restrict__ up_payload_base,
        int8_t *__restrict__ shared_gate_weights,
        int8_t *__restrict__ shared_up_weights,
        int block,
        int N,
        int column_base,
        int lane)
    {
        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<13>::payload_bytes;
        const int local_column = lane >> 2;
        const int lookup = lane & 3;
        const int column = column_base + local_column;
        uint64_t signed_gate_grid = 0;
        uint64_t signed_up_grid = 0;
        if (column < N)
        {
            const size_t linear =
                static_cast<size_t>(block) * N + column;
            const uint8_t *gate_payload =
                gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload =
                up_payload_base + linear * kPayloadBytes;
            const uint8_t gate_qh = gate_payload[4];
            const uint8_t up_qh = up_payload[4];
            const int gate_index =
                static_cast<int>(gate_payload[lookup]) |
                (static_cast<int>((gate_qh >> (2 * lookup)) & 0x3u) << 8);
            const int up_index =
                static_cast<int>(up_payload[lookup]) |
                (static_cast<int>((up_qh >> (2 * lookup)) & 0x3u) << 8);

            /*
             * Keep these adjacent and ahead of all dependent sign arithmetic.
             * nvcc can therefore issue both independent read-only requests
             * before either result reaches its packed-byte data path.
             */
            const uint64_t gate_grid =
                llaminar2::cuda_native_vnni::iq2_grid_lookup<13>(gate_index);
            const uint64_t up_grid =
                llaminar2::cuda_native_vnni::iq2_grid_lookup<13>(up_index);
            const uint8_t gate_signs = gate_payload[5 + lookup];
            const uint8_t up_signs = up_payload[5 + lookup];

            const uint32_t gate_low =
                llaminar2::cuda_native_vnni::iq_apply_signs_4(
                    static_cast<uint32_t>(gate_grid),
                    gate_signs & 0x0Fu);
            const uint32_t gate_high =
                llaminar2::cuda_native_vnni::iq_apply_signs_4(
                    static_cast<uint32_t>(gate_grid >> 32),
                    gate_signs >> 4);
            const uint32_t up_low =
                llaminar2::cuda_native_vnni::iq_apply_signs_4(
                    static_cast<uint32_t>(up_grid),
                    up_signs & 0x0Fu);
            const uint32_t up_high =
                llaminar2::cuda_native_vnni::iq_apply_signs_4(
                    static_cast<uint32_t>(up_grid >> 32),
                    up_signs >> 4);
            signed_gate_grid =
                static_cast<uint64_t>(gate_low) |
                (static_cast<uint64_t>(gate_high) << 32);
            signed_up_grid =
                static_cast<uint64_t>(up_low) |
                (static_cast<uint64_t>(up_high) << 32);
        }

        const size_t fragment_offset =
            static_cast<size_t>(local_column) * kQuantBlock +
            lookup * sizeof(uint64_t);
        *reinterpret_cast<uint64_t *>(
            shared_gate_weights + fragment_offset) = signed_gate_grid;
        *reinterpret_cast<uint64_t *>(
            shared_up_weights + fragment_offset) = signed_up_grid;
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
            const int expert_tiles =
                (count + kRows - 1) / kRows;
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
    template <uint8_t CodebookId, int ColumnsPerBlock>
    __global__ __launch_bounds__(
        GroupedImmaGeometry<ColumnsPerBlock>::projection_threads,
        GroupedImmaGeometry<ColumnsPerBlock>::
            projection_minimum_blocks_per_sm)
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
        using Geometry = GroupedImmaGeometry<ColumnsPerBlock>;
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
            (static_cast<int>(blockIdx.x) * Geometry::projection_warps + warp) *
            kColumnsPerWarp;

        constexpr bool kStageIQ4Metadata =
            CodebookId == 4 && ColumnsPerBlock == 32;
        constexpr int kScaleColumns =
            Geometry::projection_warps * kColumnsPerWarp;
        constexpr int kOperandBytes =
            kRows * kQuantBlock +
            Geometry::projection_warps * kColumnsPerWarp * kQuantBlock;
        constexpr int kStagedMetadataWords =
            kStageIQ4Metadata ? kScaleColumns : 0;
        static_assert((kOperandBytes % sizeof(uint32_t)) == 0);
        __shared__ __align__(16) uint32_t shared_storage[
            kOperandBytes / sizeof(uint32_t) + kStagedMetadataWords];
        auto *shared_bytes = reinterpret_cast<int8_t *>(shared_storage);
        int8_t *shared_a = shared_bytes;
        int8_t *shared_b =
            shared_bytes + kRows * kQuantBlock +
            warp * kColumnsPerWarp * kQuantBlock;
        uint32_t *shared_weight_scales =
            shared_storage + kOperandBytes / sizeof(uint32_t);

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
                    const int local_row =
                        static_cast<int>(threadIdx.x) >> 1;
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
                if constexpr (kStageIQ4Metadata)
                {
                    if (lane < kColumnsPerWarp)
                    {
                        const int column = column_base + lane;
                        uint32_t scale_bits = 0;
                        if (column < N)
                        {
                            const size_t linear =
                                static_cast<size_t>(block) * N + column;
                            scale_bits = weight_scales[linear];
                        }
                        shared_weight_scales[
                            warp * kColumnsPerWarp + lane] = scale_bits;
                    }
                }
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
                    const float activation_scale = scales_A[
                        static_cast<size_t>(grouped_row) * blocks_per_row +
                        block];
                    if constexpr (kStageIQ4Metadata)
                    {
                        const int scale_index =
                            warp * kColumnsPerWarp +
                            mmaFragmentColumn(lane, element);
                        partial[element] = __fadd_rn(
                            partial[element],
                            contributionFromStagedSymmetricMma<CodebookId>(
                                dot_lo[element],
                                activation_scale,
                                static_cast<uint16_t>(
                                    shared_weight_scales[scale_index])));
                    }
                    else
                    {
                        const size_t linear =
                            static_cast<size_t>(block) * N + column;
                        const uint8_t *payload =
                            descriptor.payload + linear * kPayloadBytes;
                        partial[element] = __fadd_rn(
                            partial[element],
                            contributionFromMma<CodebookId>(
                                shared_a + local_row * kQuantBlock,
                                dot_lo[element],
                                dot_hi[element],
                                payload,
                                weight_scales,
                                weight_mins,
                                weight_emins,
                                linear,
                                activation_scale));
                    }
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
    template <uint8_t CodebookId, int ColumnsPerBlock>
    __global__ __launch_bounds__(
        ParallelGateUpGeometry<ColumnsPerBlock>::gate_up_threads,
        ParallelGateUpGeometry<ColumnsPerBlock>::gate_up_minimum_blocks_per_sm)
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
        using Geometry = ParallelGateUpGeometry<ColumnsPerBlock>;
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
        const bool is_up_projection =
            warp >= Geometry::Projection::projection_warps;
        const int projection_warp = is_up_projection
                                        ? warp -
                                              Geometry::Projection::
                                                  projection_warps
                                        : warp;
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
            (static_cast<int>(blockIdx.x) *
                 Geometry::Projection::projection_warps +
             projection_warp) *
            kColumnsPerWarp;

        static_assert(
            (Geometry::gate_up_shared_bytes % sizeof(uint32_t)) == 0);
        __shared__ __align__(16) uint32_t shared_storage[
            Geometry::gate_up_shared_bytes / sizeof(uint32_t)];
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
                static_cast<int>(blockIdx.x) * ColumnsPerBlock + local_column;
            if (local_row < active_rows && column < N)
            {
                shared_projections[
                    (projection * kRows + local_row) * ColumnsPerBlock +
                    local_column] = total[element];
            }
        }
        __syncthreads();

        constexpr int kQuantBlocksPerCta = ColumnsPerBlock / kQuantBlock;
        const int output_blocks_per_row =
            (N + kQuantBlock - 1) / kQuantBlock;
        const int quantization_tasks = active_rows * kQuantBlocksPerCta;
        for (int task = warp; task < quantization_tasks;
             task += Geometry::gate_up_warps)
        {
            const int local_row = task / kQuantBlocksPerCta;
            const int local_quant_block = task % kQuantBlocksPerCta;
            const int quant_column =
                static_cast<int>(blockIdx.x) * ColumnsPerBlock +
                local_quant_block * kQuantBlock + lane;
            const bool quant_column_active = quant_column < N;
            const int fragment_index =
                local_row * ColumnsPerBlock +
                local_quant_block * kQuantBlock + lane;
            const float gate = quant_column_active
                                   ? shared_projections[fragment_index]
                                   : 0.0f;
            const float up = quant_column_active
                                 ? shared_projections[
                                       kRows * ColumnsPerBlock + fragment_index]
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
                        static_cast<int>(blockIdx.x) * kQuantBlocksPerCta +
                        local_quant_block] = scale;
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

    /**
     * @brief Execute the proven sixteen-row paired gate/up schedule.
     *
     * A warp owns eight output columns for both projections. It decodes the two
     * B fragments into disjoint shared ranges, loads the common A fragment once,
     * and evaluates the fixed partition tree without an intervening
     * publication. Only the exact SwiGLU value enters shared memory for the
     * unchanged 32-column blockwise quantizer.
     */
    template <uint8_t CodebookId, int ColumnsPerBlock>
    __global__ __launch_bounds__(
        PairedGateUpGeometry<ColumnsPerBlock>::gate_up_threads,
        PairedGateUpGeometry<ColumnsPerBlock>::
            gate_up_minimum_blocks_per_sm)
    void groupedImmaPairedGateUpSwiGluKernel(
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
        using Geometry = PairedGateUpGeometry<ColumnsPerBlock>;
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
        if (gate_descriptors[expert].codebook_id != CodebookId)
            return;

        const auto &gate_descriptor = gate_descriptors[expert];
        const auto &up_descriptor = up_descriptors[expert];
        if (!descriptorMatches<CodebookId>(gate_descriptor, N, K) ||
            !descriptorMatches<CodebookId>(up_descriptor, N, K) ||
            k_partitions <= 0)
        {
            failFastInvalidGroupedImmaState();
            return;
        }

        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int active_rows = min(kRows, count - local_first_row);
        const int grouped_row_base = group_offset + local_first_row;
        const int column_base =
            (static_cast<int>(blockIdx.x) * Geometry::gate_up_warps + warp) *
            kColumnsPerWarp;

        static_assert(
            (Geometry::gate_up_shared_bytes % sizeof(uint32_t)) == 0);
        constexpr int kScaleColumns =
            Geometry::gate_up_warps * kColumnsPerWarp;
        constexpr bool kStageIQ2SMetadata =
            CodebookId == 13 && ColumnsPerBlock == 32;
        constexpr int kStagedMetadataWords =
            kStageIQ2SMetadata ? 2 * kScaleColumns + kRows : 0;
        __shared__ __align__(16) uint32_t shared_storage[
            Geometry::gate_up_shared_bytes / sizeof(uint32_t) +
            kStagedMetadataWords];
        auto *shared_bytes = reinterpret_cast<int8_t *>(shared_storage);
        int8_t *shared_a = shared_bytes;
        constexpr int kWarpWeightBytes = kColumnsPerWarp * kQuantBlock;
        int8_t *shared_gate_b =
            shared_bytes + kRows * kQuantBlock + warp * kWarpWeightBytes;
        int8_t *shared_up_b =
            shared_bytes + kRows * kQuantBlock +
            Geometry::gate_up_warps * kWarpWeightBytes +
            warp * kWarpWeightBytes;

        uint32_t *shared_gate_scale_pairs =
            shared_storage +
            Geometry::gate_up_shared_bytes / sizeof(uint32_t);
        uint32_t *shared_up_scale_pairs =
            shared_gate_scale_pairs + kScaleColumns;
        float *shared_activation_scales = reinterpret_cast<float *>(
            shared_up_scale_pairs + kScaleColumns);

        constexpr int kPayloadBytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CodebookId>::
                payload_bytes;
        const uint8_t *gate_payload_base = gate_descriptor.payload;
        const uint8_t *up_payload_base = up_descriptor.payload;
        const auto *gate_scales =
            static_cast<const uint16_t *>(gate_descriptor.scales);
        const auto *gate_mins =
            static_cast<const uint16_t *>(gate_descriptor.mins);
        const auto *gate_emins =
            static_cast<const uint32_t *>(gate_descriptor.emins);
        const auto *up_scales =
            static_cast<const uint16_t *>(up_descriptor.scales);
        const auto *up_mins =
            static_cast<const uint16_t *>(up_descriptor.mins);
        const auto *up_emins =
            static_cast<const uint32_t *>(up_descriptor.emins);
        const int blocks_per_row = K / kQuantBlock;
        const int blocks_per_partition =
            (blocks_per_row + k_partitions - 1) / k_partitions;
        float gate_total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float up_total[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int partition = 0; partition < k_partitions; ++partition)
        {
            float gate_partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int block_begin = partition * blocks_per_partition;
            const int block_end = min(
                blocks_per_row,
                block_begin + blocks_per_partition);

            for (int block = block_begin; block < block_end; ++block)
            {
                if (threadIdx.x < 32)
                {
                    const int local_row =
                        static_cast<int>(threadIdx.x) >> 1;
                    const int half = static_cast<int>(threadIdx.x) & 1;
                    int4 activation = make_int4(0, 0, 0, 0);
                    if (local_row < active_rows)
                    {
                        const int grouped_row =
                            grouped_row_base + local_row;
                        activation = *reinterpret_cast<const int4 *>(
                            A_int8 +
                            static_cast<size_t>(grouped_row) * K +
                            block * kQuantBlock + half * sizeof(int4));
                    }
                    *reinterpret_cast<int4 *>(
                        shared_a + local_row * kQuantBlock +
                        half * sizeof(int4)) = activation;
                }

                if constexpr (CodebookId == 13)
                {
                    decodePairedIQ2SWeightFragments(
                        gate_payload_base,
                        up_payload_base,
                        shared_gate_b,
                        shared_up_b,
                        block,
                        N,
                        column_base,
                        lane);
                }
                else
                {
                    decodeMmaWeightFragment<CodebookId>(
                        gate_descriptor,
                        shared_gate_b,
                        block,
                        N,
                        column_base,
                        lane);
                    decodeMmaWeightFragment<CodebookId>(
                        up_descriptor,
                        shared_up_b,
                        block,
                        N,
                        column_base,
                        lane);
                }

                if constexpr (kStageIQ2SMetadata)
                {
                    if (lane < kColumnsPerWarp)
                    {
                        const int column = column_base + lane;
                        uint32_t gate_scale_pair = 0;
                        uint32_t up_scale_pair = 0;
                        if (column < N)
                        {
                            const size_t linear =
                                static_cast<size_t>(block) * N + column;
                            gate_scale_pair =
                                static_cast<uint32_t>(gate_scales[linear]) |
                                (static_cast<uint32_t>(gate_mins[linear])
                                 << 16);
                            up_scale_pair =
                                static_cast<uint32_t>(up_scales[linear]) |
                                (static_cast<uint32_t>(up_mins[linear]) << 16);
                        }
                        const int scale_index =
                            warp * kColumnsPerWarp + lane;
                        shared_gate_scale_pairs[scale_index] = gate_scale_pair;
                        shared_up_scale_pairs[scale_index] = up_scale_pair;
                    }
                    if (threadIdx.x < kRows)
                    {
                        const int local_row =
                            static_cast<int>(threadIdx.x);
                        float activation_scale = 0.0f;
                        if (local_row < active_rows)
                        {
                            activation_scale = scales_A[
                                static_cast<size_t>(
                                    grouped_row_base + local_row) *
                                    blocks_per_row +
                                block];
                        }
                        shared_activation_scales[local_row] = activation_scale;
                    }
                }
                __syncthreads();

                uint32_t a_fragment[4];
                uint32_t gate_b_fragment[2];
                uint32_t up_b_fragment[2];
                loadMmaA(
                    a_fragment,
                    reinterpret_cast<const int *>(shared_a),
                    lane);
                loadMmaB(
                    gate_b_fragment,
                    reinterpret_cast<const int *>(shared_gate_b),
                    lane);
                loadMmaB(
                    up_b_fragment,
                    reinterpret_cast<const int *>(shared_up_b),
                    lane);

                int32_t gate_dot_lo[4] = {0, 0, 0, 0};
                int32_t gate_dot_hi[4] = {0, 0, 0, 0};
                int32_t up_dot_lo[4] = {0, 0, 0, 0};
                int32_t up_dot_hi[4] = {0, 0, 0, 0};
                if constexpr (llaminar2::cuda_native_vnni::
                                  CodebookTraits<CodebookId>::is_dual_scale)
                {
                    const uint32_t gate_b_lo[2] = {
                        gate_b_fragment[0], 0u};
                    const uint32_t gate_b_hi[2] = {
                        0u, gate_b_fragment[1]};
                    const uint32_t up_b_lo[2] = {
                        up_b_fragment[0], 0u};
                    const uint32_t up_b_hi[2] = {
                        0u, up_b_fragment[1]};
                    mmaM16N8K32(gate_dot_lo, a_fragment, gate_b_lo);
                    mmaM16N8K32(gate_dot_hi, a_fragment, gate_b_hi);
                    mmaM16N8K32(up_dot_lo, a_fragment, up_b_lo);
                    mmaM16N8K32(up_dot_hi, a_fragment, up_b_hi);
                }
                else
                {
                    mmaM16N8K32(
                        gate_dot_lo,
                        a_fragment,
                        gate_b_fragment);
                    mmaM16N8K32(
                        up_dot_lo,
                        a_fragment,
                        up_b_fragment);
                }

#pragma unroll
                for (int element = 0; element < 4; ++element)
                {
                    const int local_row = mmaFragmentRow(lane, element);
                    const int column =
                        column_base + mmaFragmentColumn(lane, element);
                    if (local_row >= active_rows || column >= N)
                        continue;

                    if constexpr (kStageIQ2SMetadata)
                    {
                        const int scale_index =
                            warp * kColumnsPerWarp +
                            mmaFragmentColumn(lane, element);
                        const float activation_scale =
                            shared_activation_scales[local_row];
                        gate_partial[element] = __fadd_rn(
                            gate_partial[element],
                            contributionFromStagedDualScaleMma<CodebookId>(
                                gate_dot_lo[element],
                                gate_dot_hi[element],
                                activation_scale,
                                shared_gate_scale_pairs[scale_index]));
                        up_partial[element] = __fadd_rn(
                            up_partial[element],
                            contributionFromStagedDualScaleMma<CodebookId>(
                                up_dot_lo[element],
                                up_dot_hi[element],
                                activation_scale,
                                shared_up_scale_pairs[scale_index]));
                    }
                    else
                    {
                        const int grouped_row = grouped_row_base + local_row;
                        const size_t linear =
                            static_cast<size_t>(block) * N + column;
                        const float activation_scale = scales_A[
                            static_cast<size_t>(grouped_row) * blocks_per_row +
                            block];
                        gate_partial[element] = __fadd_rn(
                            gate_partial[element],
                            contributionFromMma<CodebookId>(
                                shared_a + local_row * kQuantBlock,
                                gate_dot_lo[element],
                                gate_dot_hi[element],
                                gate_payload_base + linear * kPayloadBytes,
                                gate_scales,
                                gate_mins,
                                gate_emins,
                                linear,
                                activation_scale));
                        up_partial[element] = __fadd_rn(
                            up_partial[element],
                            contributionFromMma<CodebookId>(
                                shared_a + local_row * kQuantBlock,
                                up_dot_lo[element],
                                up_dot_hi[element],
                                up_payload_base + linear * kPayloadBytes,
                                up_scales,
                                up_mins,
                                up_emins,
                                linear,
                                activation_scale));
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (int element = 0; element < 4; ++element)
            {
                gate_total[element] = __fadd_rn(
                    gate_total[element],
                    gate_partial[element]);
                up_total[element] = __fadd_rn(
                    up_total[element],
                    up_partial[element]);
            }
        }

        float *shared_values = reinterpret_cast<float *>(shared_storage);
#pragma unroll
        for (int element = 0; element < 4; ++element)
        {
            const int local_row = mmaFragmentRow(lane, element);
            const int local_column =
                warp * kColumnsPerWarp +
                mmaFragmentColumn(lane, element);
            const int column =
                static_cast<int>(blockIdx.x) * ColumnsPerBlock + local_column;
            if (local_row < active_rows && column < N)
            {
                shared_values[local_row * ColumnsPerBlock + local_column] =
                    groupedPrefillSilu(gate_total[element]) *
                    up_total[element];
            }
        }
        __syncthreads();

        constexpr int kQuantBlocksPerCta = ColumnsPerBlock / kQuantBlock;
        const int output_blocks_per_row =
            (N + kQuantBlock - 1) / kQuantBlock;
        const int quantization_tasks = active_rows * kQuantBlocksPerCta;
        for (int task = warp; task < quantization_tasks;
             task += Geometry::gate_up_warps)
        {
            const int local_row = task / kQuantBlocksPerCta;
            const int local_quant_block = task % kQuantBlocksPerCta;
            const int quant_column =
                static_cast<int>(blockIdx.x) * ColumnsPerBlock +
                local_quant_block * kQuantBlock + lane;
            const bool quant_column_active = quant_column < N;
            const int fragment_index =
                local_row * ColumnsPerBlock +
                local_quant_block * kQuantBlock + lane;
            const float value = quant_column_active
                                    ? shared_values[fragment_index]
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
                        static_cast<int>(blockIdx.x) * kQuantBlocksPerCta +
                        local_quant_block] = scale;
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
        int columns_per_block,
        dim3 &grid)
    {
        constexpr unsigned int kGridDimensionLimit = 65535u;
        if (N <= 0 || directory_entries <= 0 || columns_per_block <= 0)
            return false;
        const unsigned int rows =
            static_cast<unsigned int>(directory_entries);
        const unsigned int rows_y = std::min(rows, kGridDimensionLimit);
        const unsigned int rows_z = (rows + rows_y - 1u) / rows_y;
        if (rows_z == 0 || rows_z > kGridDimensionLimit)
            return false;
        grid = dim3(
            static_cast<unsigned int>((N + columns_per_block - 1) /
                                      columns_per_block),
            rows_y,
            rows_z);
        return grid.x > 0;
    }

    /** Launch one codebook specialization when its table mask bit is present. */
    template <uint8_t CodebookId, int ColumnsPerBlock>
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
        groupedImmaProjectionKernel<CodebookId, ColumnsPerBlock><<<
            grid,
            GroupedImmaGeometry<ColumnsPerBlock>::projection_threads,
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
    template <uint8_t CodebookId, int ColumnsPerBlock>
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
        groupedImmaGateUpSwiGluKernel<CodebookId, ColumnsPerBlock><<<
            grid,
            ParallelGateUpGeometry<ColumnsPerBlock>::gate_up_threads,
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

    /** Launch one paired gate/up specialization when its mask bit exists. */
    template <uint8_t CodebookId, int ColumnsPerBlock>
    bool launchCodebookPairedGateUpSwiGlu(
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
        groupedImmaPairedGateUpSwiGluKernel<
            CodebookId,
            ColumnsPerBlock><<<
            grid,
            PairedGateUpGeometry<ColumnsPerBlock>::gate_up_threads,
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

    /** Dispatch every codebook in a mixed descriptor table at one CTA width. */
    template <int ColumnsPerBlock>
    bool launchGroupedImmaProjectionTable(
        uint32_t codebook_mask,
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
        int k_partitions)
    {
        dim3 grid;
        if (!makeProjectionGrid(N, directory_entries, ColumnsPerBlock, grid))
            return false;

        bool launched = false;
#define LAUNCH_CODEBOOK(CB)                                                     \
    do                                                                           \
    {                                                                            \
        if (!launchCodebookProjection<CB, ColumnsPerBlock>(                      \
                codebook_mask, grid, stream, A_int8, scales_A, descriptors,     \
                group_counts, group_offsets, directory, partition_weights,      \
                output, directory_entries, num_experts, total_slots, N, K,      \
                k_partitions, launched))                                         \
        {                                                                        \
            return false;                                                        \
        }                                                                        \
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

    /** Dispatch every fused gate/up codebook at one capture-time CTA width. */
    template <int ColumnsPerBlock>
    bool launchGroupedImmaGateUpTable(
        uint32_t codebook_mask,
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
        int k_partitions)
    {
        dim3 grid;
        if (!makeProjectionGrid(N, directory_entries, ColumnsPerBlock, grid))
            return false;

        bool launched = false;
#define LAUNCH_GATE_UP_CODEBOOK(CB)                                             \
    do                                                                           \
    {                                                                            \
        if (!launchCodebookGateUpSwiGlu<CB, ColumnsPerBlock>(                    \
                codebook_mask, grid, stream, A_int8, scales_A,                  \
                gate_descriptors, up_descriptors, group_counts, group_offsets,  \
                directory, swiglu_int8, swiglu_scales, directory_entries,       \
                num_experts, total_slots, N, K, k_partitions, launched))         \
        {                                                                        \
            return false;                                                        \
        }                                                                        \
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

    /** Dispatch every paired gate/up codebook at one capture-time CTA width. */
    template <int ColumnsPerBlock>
    bool launchGroupedImmaPairedGateUpTable(
        uint32_t codebook_mask,
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
        int k_partitions)
    {
        dim3 grid;
        if (!makeProjectionGrid(N, directory_entries, ColumnsPerBlock, grid))
            return false;

        bool launched = false;
#define LAUNCH_PAIRED_GATE_UP_CODEBOOK(CB)                                      \
    do                                                                           \
    {                                                                            \
        if (!launchCodebookPairedGateUpSwiGlu<CB, ColumnsPerBlock>(              \
                codebook_mask, grid, stream, A_int8, scales_A,                  \
                gate_descriptors, up_descriptors, group_counts, group_offsets,  \
                directory, swiglu_int8, swiglu_scales, directory_entries,       \
                num_experts, total_slots, N, K, k_partitions, launched))         \
        {                                                                        \
            return false;                                                        \
        }                                                                        \
    } while (0)

        LAUNCH_PAIRED_GATE_UP_CODEBOOK(0);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(4);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(5);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(6);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(7);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(8);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(9);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(10);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(11);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(12);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(13);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(14);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(15);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(16);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(17);
        LAUNCH_PAIRED_GATE_UP_CODEBOOK(19);

#undef LAUNCH_PAIRED_GATE_UP_CODEBOOK
        return launched;
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
    llaminar2::cuda::moe::GroupedImmaColumns columns,
    int device_idx,
    void *stream)
{
    if (!d_A_int8 || !d_scales_A || !d_desc_table ||
        !d_group_counts || !d_group_offsets || !d_directory || !d_output ||
        !stream || directory_entries <= 0 || num_experts <= 0 ||
        num_experts > llaminar2::cuda::moe::kGroupedImmaMaximumExperts ||
        total_slots <= 0 || N <= 0 || K <= 0 || (K % kQuantBlock) != 0 ||
        codebook_mask == 0 || (codebook_mask & ~kSupportedCodebookMask) != 0 ||
        k_partitions <= 0 ||
        !llaminar2::cuda::moe::validGroupedImmaDownColumns(columns))
    {
        return false;
    }
    if (cudaSetDevice(device_idx) != cudaSuccess)
        return false;

    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    using llaminar2::cuda::moe::GroupedImmaColumns;
    switch (columns)
    {
    case GroupedImmaColumns::Columns32:
        return launchGroupedImmaProjectionTable<32>(
            codebook_mask, cuda_stream, d_A_int8, d_scales_A, d_desc_table,
            d_group_counts, d_group_offsets, d_directory, d_partition_weights,
            d_output, directory_entries, num_experts, total_slots, N, K,
            k_partitions);
    case GroupedImmaColumns::Columns64:
        return launchGroupedImmaProjectionTable<64>(
            codebook_mask, cuda_stream, d_A_int8, d_scales_A, d_desc_table,
            d_group_counts, d_group_offsets, d_directory, d_partition_weights,
            d_output, directory_entries, num_experts, total_slots, N, K,
            k_partitions);
    case GroupedImmaColumns::Columns128:
        return launchGroupedImmaProjectionTable<128>(
            codebook_mask, cuda_stream, d_A_int8, d_scales_A, d_desc_table,
            d_group_counts, d_group_offsets, d_directory, d_partition_weights,
            d_output, directory_entries, num_experts, total_slots, N, K,
            k_partitions);
    case GroupedImmaColumns::Columns256:
        return launchGroupedImmaProjectionTable<256>(
            codebook_mask, cuda_stream, d_A_int8, d_scales_A, d_desc_table,
            d_group_counts, d_group_offsets, d_directory, d_partition_weights,
            d_output, directory_entries, num_experts, total_slots, N, K,
            k_partitions);
    }
    return false;
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
    llaminar2::cuda::moe::GroupedImmaColumns columns,
    llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule,
    int device_idx,
    void *stream)
{
    if (!d_A_int8 || !d_scales_A || !d_gate_desc_table ||
        !d_up_desc_table || !d_group_counts || !d_group_offsets ||
        !d_directory || !d_swiglu_int8 || !d_swiglu_scales || !stream ||
        directory_entries <= 0 || num_experts <= 0 ||
        num_experts > llaminar2::cuda::moe::kGroupedImmaMaximumExperts ||
        total_slots <= 0 || N <= 0 || K <= 0 ||
        (N % kQuantBlock) != 0 || (K % kQuantBlock) != 0 ||
        codebook_mask == 0 ||
        (codebook_mask & ~kSupportedCodebookMask) != 0 ||
        k_partitions <= 0 ||
        !llaminar2::cuda::moe::validGroupedImmaGateUpColumns(columns) ||
        !llaminar2::cuda::moe::validGroupedImmaGateUpSchedule(schedule))
    {
        return false;
    }
    if (cudaSetDevice(device_idx) != cudaSuccess)
        return false;

    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    using llaminar2::cuda::moe::GroupedImmaColumns;
    using llaminar2::cuda::moe::GroupedImmaGateUpSchedule;
    const bool paired =
        schedule == GroupedImmaGateUpSchedule::PairedProjections;
    switch (columns)
    {
    case GroupedImmaColumns::Columns32:
        return paired
                   ? launchGroupedImmaPairedGateUpTable<32>(
                         codebook_mask, cuda_stream, d_A_int8, d_scales_A,
                         d_gate_desc_table, d_up_desc_table, d_group_counts,
                         d_group_offsets, d_directory, d_swiglu_int8,
                         d_swiglu_scales, directory_entries, num_experts,
                         total_slots, N, K, k_partitions)
                   : launchGroupedImmaGateUpTable<32>(
                         codebook_mask, cuda_stream, d_A_int8, d_scales_A,
                         d_gate_desc_table, d_up_desc_table, d_group_counts,
                         d_group_offsets, d_directory, d_swiglu_int8,
                         d_swiglu_scales, directory_entries, num_experts,
                         total_slots, N, K, k_partitions);
    case GroupedImmaColumns::Columns64:
        return paired
                   ? launchGroupedImmaPairedGateUpTable<64>(
                         codebook_mask, cuda_stream, d_A_int8, d_scales_A,
                         d_gate_desc_table, d_up_desc_table, d_group_counts,
                         d_group_offsets, d_directory, d_swiglu_int8,
                         d_swiglu_scales, directory_entries, num_experts,
                         total_slots, N, K, k_partitions)
                   : launchGroupedImmaGateUpTable<64>(
                         codebook_mask, cuda_stream, d_A_int8, d_scales_A,
                         d_gate_desc_table, d_up_desc_table, d_group_counts,
                         d_group_offsets, d_directory, d_swiglu_int8,
                         d_swiglu_scales, directory_entries, num_experts,
                         total_slots, N, K, k_partitions);
    case GroupedImmaColumns::Columns128:
        return paired
                   ? launchGroupedImmaPairedGateUpTable<128>(
                         codebook_mask, cuda_stream, d_A_int8, d_scales_A,
                         d_gate_desc_table, d_up_desc_table, d_group_counts,
                         d_group_offsets, d_directory, d_swiglu_int8,
                         d_swiglu_scales, directory_entries, num_experts,
                         total_slots, N, K, k_partitions)
                   : launchGroupedImmaGateUpTable<128>(
                         codebook_mask, cuda_stream, d_A_int8, d_scales_A,
                         d_gate_desc_table, d_up_desc_table, d_group_counts,
                         d_group_offsets, d_directory, d_swiglu_int8,
                         d_swiglu_scales, directory_entries, num_experts,
                         total_slots, N, K, k_partitions);
    case GroupedImmaColumns::Columns256:
        return false;
    }
    return false;
}

namespace
{
    /** Query compiler resources for one exact IMMA specialization. */
    template <uint8_t CodebookId, int ColumnsPerBlock>
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
                groupedImmaProjectionKernel<CodebookId, ColumnsPerBlock>) !=
            cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                groupedImmaProjectionKernel<CodebookId, ColumnsPerBlock>,
                GroupedImmaGeometry<ColumnsPerBlock>::projection_threads,
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
    template <uint8_t CodebookId, int ColumnsPerBlock>
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
                groupedImmaGateUpSwiGluKernel<CodebookId, ColumnsPerBlock>) !=
            cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                groupedImmaGateUpSwiGluKernel<CodebookId, ColumnsPerBlock>,
                ParallelGateUpGeometry<ColumnsPerBlock>::gate_up_threads,
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

    /** Query compiler resources for one paired gate/up specialization. */
    template <uint8_t CodebookId, int ColumnsPerBlock>
    bool queryGroupedImmaPairedGateUpKernelResources(
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(
                &attributes,
                groupedImmaPairedGateUpSwiGluKernel<
                    CodebookId,
                    ColumnsPerBlock>) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                groupedImmaPairedGateUpSwiGluKernel<
                    CodebookId,
                    ColumnsPerBlock>,
                PairedGateUpGeometry<ColumnsPerBlock>::gate_up_threads,
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

    /** Resolve one down-projection codebook at a compile-time geometry. */
    template <int ColumnsPerBlock>
    bool queryGroupedImmaCodebookResources(
        uint8_t codebook_id,
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
#define QUERY_CODEBOOK(CB)                                                      \
    case CB:                                                                    \
        return queryGroupedImmaKernelResources<CB, ColumnsPerBlock>(            \
            registers_per_thread, local_memory_bytes_per_thread,                \
            static_shared_memory_bytes, max_threads_per_block,                  \
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

    /** Resolve one fused gate/up codebook at a compile-time geometry. */
    template <int ColumnsPerBlock, bool Paired>
    bool queryGroupedImmaGateUpCodebookResources(
        uint8_t codebook_id,
        int *registers_per_thread,
        std::size_t *local_memory_bytes_per_thread,
        std::size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm)
    {
#define QUERY_GATE_UP_CODEBOOK(CB)                                               \
    case CB:                                                                     \
        if constexpr (Paired)                                                    \
            return queryGroupedImmaPairedGateUpKernelResources<                  \
                CB, ColumnsPerBlock>(                                            \
                registers_per_thread, local_memory_bytes_per_thread,             \
                static_shared_memory_bytes, max_threads_per_block,               \
                max_active_blocks_per_sm);                                       \
        else                                                                     \
            return queryGroupedImmaGateUpKernelResources<CB, ColumnsPerBlock>(   \
                registers_per_thread, local_memory_bytes_per_thread,             \
                static_shared_memory_bytes, max_threads_per_block,               \
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
}

extern "C" bool cudaMoEGroupedImma_queryKernelResources(
    uint8_t codebook_id,
    llaminar2::cuda::moe::GroupedImmaColumns columns,
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

    using llaminar2::cuda::moe::GroupedImmaColumns;
    switch (columns)
    {
    case GroupedImmaColumns::Columns32:
        return queryGroupedImmaCodebookResources<32>(
            codebook_id, registers_per_thread, local_memory_bytes_per_thread,
            static_shared_memory_bytes, max_threads_per_block,
            max_active_blocks_per_sm);
    case GroupedImmaColumns::Columns64:
        return queryGroupedImmaCodebookResources<64>(
            codebook_id, registers_per_thread, local_memory_bytes_per_thread,
            static_shared_memory_bytes, max_threads_per_block,
            max_active_blocks_per_sm);
    case GroupedImmaColumns::Columns128:
        return queryGroupedImmaCodebookResources<128>(
            codebook_id, registers_per_thread, local_memory_bytes_per_thread,
            static_shared_memory_bytes, max_threads_per_block,
            max_active_blocks_per_sm);
    case GroupedImmaColumns::Columns256:
        return queryGroupedImmaCodebookResources<256>(
            codebook_id, registers_per_thread, local_memory_bytes_per_thread,
            static_shared_memory_bytes, max_threads_per_block,
            max_active_blocks_per_sm);
    }
    return false;
}

extern "C" bool cudaMoEGroupedImma_queryGateUpKernelResources(
    uint8_t codebook_id,
    llaminar2::cuda::moe::GroupedImmaColumns columns,
    llaminar2::cuda::moe::GroupedImmaGateUpSchedule schedule,
    int *registers_per_thread,
    std::size_t *local_memory_bytes_per_thread,
    std::size_t *static_shared_memory_bytes,
    int *max_threads_per_block,
    int *max_active_blocks_per_sm)
{
    if (!registers_per_thread || !local_memory_bytes_per_thread ||
        !static_shared_memory_bytes || !max_threads_per_block ||
        !max_active_blocks_per_sm ||
        !llaminar2::cuda::moe::validGroupedImmaGateUpSchedule(schedule))
    {
        return false;
    }

    using llaminar2::cuda::moe::GroupedImmaColumns;
    using llaminar2::cuda::moe::GroupedImmaGateUpSchedule;
    const bool paired =
        schedule == GroupedImmaGateUpSchedule::PairedProjections;
    switch (columns)
    {
    case GroupedImmaColumns::Columns32:
        return paired
                   ? queryGroupedImmaGateUpCodebookResources<32, true>(
                         codebook_id, registers_per_thread,
                         local_memory_bytes_per_thread,
                         static_shared_memory_bytes, max_threads_per_block,
                         max_active_blocks_per_sm)
                   : queryGroupedImmaGateUpCodebookResources<32, false>(
                         codebook_id, registers_per_thread,
                         local_memory_bytes_per_thread,
                         static_shared_memory_bytes, max_threads_per_block,
                         max_active_blocks_per_sm);
    case GroupedImmaColumns::Columns64:
        return paired
                   ? queryGroupedImmaGateUpCodebookResources<64, true>(
                         codebook_id, registers_per_thread,
                         local_memory_bytes_per_thread,
                         static_shared_memory_bytes, max_threads_per_block,
                         max_active_blocks_per_sm)
                   : queryGroupedImmaGateUpCodebookResources<64, false>(
                         codebook_id, registers_per_thread,
                         local_memory_bytes_per_thread,
                         static_shared_memory_bytes, max_threads_per_block,
                         max_active_blocks_per_sm);
    case GroupedImmaColumns::Columns128:
        return paired
                   ? queryGroupedImmaGateUpCodebookResources<128, true>(
                         codebook_id, registers_per_thread,
                         local_memory_bytes_per_thread,
                         static_shared_memory_bytes, max_threads_per_block,
                         max_active_blocks_per_sm)
                   : queryGroupedImmaGateUpCodebookResources<128, false>(
                         codebook_id, registers_per_thread,
                         local_memory_bytes_per_thread,
                         static_shared_memory_bytes, max_threads_per_block,
                         max_active_blocks_per_sm);
    case GroupedImmaColumns::Columns256:
        return false;
    }
    return false;
}
