/**
 * @file CUDANativeVNNIPrefillDevice.cuh
 * @brief Shared CUDA prefill device primitives and the canonical BK64 tile.
 *
 * Production launch bridges and focused candidate tests instantiate this same
 * arithmetic body. Staging changes only how immutable packed operands reach the
 * existing two shared-memory slots. It never changes persistent VRAM storage,
 * scales, MMA arithmetic, public-M1 partition boundaries or the ordered FP32 fold.
 *
 * An asynchronous slot has one copy/decode owner per packed block. That owner
 * waits for its own copies before expanding in place; the CTA publication
 * barrier then joins all decoded payload and metadata producers. Slot reuse
 * follows the same barrier after every old consumer is finished. No allocation,
 * host transfer, or stream/device synchronization occurs inside the kernel.
 * RegisterDecode remains the default until a complete resource/byte/economy
 * tournament authorizes another explicitly represented launch configuration.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "CUDANativeVNNIDecodeCommon.cuh"
#include "CUDANativeVNNIPrefillSchedule.h"

namespace llaminar2::cuda::prefill
{
    using llaminar2::cuda_native_vnni::fp16_bits_to_float;

    /// Two adjacent native 32-value blocks form one INT8 MMA K tile.
    inline constexpr int BK64 = 64;
    /// Padding retains 16-byte alignment for shared ldmatrix row addresses.
    inline constexpr int SMEM_PAD_64 = 16;
    inline constexpr int SMEM_STRIDE_64 = BK64 + SMEM_PAD_64;

    /** @brief Map one MMA accumulator element to its logical output row.
     * @param lane_id Lane in the 32-thread warp.
     * @param elem Accumulator element, in the range [0,4).
     * @return Row offset within the 16-row MMA fragment.
     */
    [[maybe_unused]] __device__ __forceinline__ int frag_row(int lane_id, int elem)
    {
        return (elem >> 1) * 8 + (lane_id >> 2);
    }

    /** @brief Map one MMA accumulator element to its logical output column.
     * @param lane_id Lane in the 32-thread warp.
     * @param elem Accumulator element, in the range [0,4).
     * @return Column offset within the eight-column MMA fragment.
     */
    [[maybe_unused]] __device__ __forceinline__ int frag_col(int lane_id, int elem)
    {
        return (lane_id & 3) * 2 + (elem & 1);
    }

    /** @brief Queue one aligned global-to-shared copy in this thread's group.
     * @param smem_dst Sixteen-byte-aligned destination with unique copy ownership.
     * @param gmem_src Sixteen-byte-aligned persistent global source.
     * @param src_size Sixteen for a valid vector, zero for a zero-filled tail.
     * Callers must commit and wait, then join producers before another thread reads.
     */
    [[maybe_unused]] __device__ __forceinline__ void cp_async_cg_16_zfill_128(
        void *smem_dst, const void *gmem_src, int src_size)
    {
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_src), "r"(src_size) : "memory");
    }

    /** @brief Commit this thread's queued copies as one asynchronous group. */
    [[maybe_unused]] __device__ __forceinline__ void cp_async_commit()
    {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    /** @brief Wait until at most N committed copy groups remain outstanding.
     * @tparam N Immediate hardware wait-group bound; zero joins all own copies.
     * This is thread-owned completion, not a CTA-wide publication barrier.
     */
    template <int N>
    __device__ __forceinline__ void cp_async_wait()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
    }

    // Named barrier: sync a subset of threads within a CTA.
    // barrier_id: 0-15 (hardware barrier slot), thread_count: participating threads.
    // Has acquire/release memory ordering (like __syncthreads but scoped to participants).
    /** @brief Join the declared CTA participants with acquire/release ordering.
     * @param barrier_id Hardware barrier slot in [0,16).
     * @param thread_count Exact number of participating threads.
     */
    [[maybe_unused]] __device__ __forceinline__ void named_bar_sync(int barrier_id, int thread_count)
    {
        asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(thread_count) : "memory");
    }

    /** @brief Read the activation fragment after shared-memory publication.
     * @param frag Receives four packed activation registers.
     * @param smem_base Published activation tile, addressed in 32-bit words.
     * @param stride_words Shared row stride, including layout padding.
     * @param lane_id Lane participating in the complete warp load.
     */
    [[maybe_unused]] __device__ __forceinline__ void load_ldmatrix_a_m16n8k32(
        uint32_t frag[4],
        const int *smem_base,
        int stride_words,
        int lane_id)
    {
#if __CUDA_ARCH__ >= 800
        const int *xs = smem_base + (lane_id % 16) * stride_words + (lane_id / 16) * 4;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
                     : "l"(xs));
#else
        (void)frag;
        (void)smem_base;
        (void)stride_words;
        (void)lane_id;
#endif
    }

    /** @brief Read the decoded weight fragment after producer publication.
     * @param frag Receives two packed weight registers.
     * @param smem_base Published decoded weight tile in 32-bit words.
     * @param stride_words Shared row stride, including layout padding.
     * @param lane_id Lane participating in the complete warp load.
     */
    [[maybe_unused]] __device__ __forceinline__ void load_ldmatrix_b_m16n8k32(
        uint32_t frag[2],
        const int *smem_base,
        int stride_words,
        int lane_id)
    {
#if __CUDA_ARCH__ >= 800
        const int *xs = smem_base + (lane_id % 8) * stride_words + ((lane_id / 8) * 4) % 8;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                     : "=r"(frag[0]), "=r"(frag[1])
                     : "l"(xs));
#else
        (void)frag;
        (void)smem_base;
        (void)stride_words;
        (void)lane_id;
#endif
    }

    /** @brief Accumulate one exact signed-INT8 MMA fragment into INT32.
     * @param D Input/output integer accumulator registers.
     * @param A Four activation fragment registers from ldmatrix.
     * @param B Two weight fragment registers from ldmatrix.
     * Floating scaling and the canonical ordered fold remain separate operations.
     */
    [[maybe_unused]] __device__ __forceinline__ void mma_m16n8k32_s8(
        int32_t D[4],
        const uint32_t A[4],
        const uint32_t B[2])
    {
#if __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%0, %1, %2, %3};\n"
            : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              "r"(B[0]), "r"(B[1]));
#else
        (void)D;
        (void)A;
        (void)B;
#endif
    }

    // =========================================================================
    // BK=64 kernel: processes 2 quantized blocks per K-tile iteration.
    // Halves the number of K-loop iterations, halving barrier and decode
    // overhead while doubling MMA work per iteration.
    // Templatized on CODEBOOK_ID for any single-scale or asymmetric format.
    // For 16-byte payloads (CB 0,4,5): vectorized int4 load + specialized decode.
    // Other payloads use their registered decode_groups<CB>() implementation.
    // Asymmetric formats (CB 5,7): adds min correction via sum_A.
    // B payload is decoded directly from global → registers → smem_B
    // (no staging buffer), saving 8KB smem and eliminating extra traffic.
    // =========================================================================
    // Occupancy hint: BM=128 (8 warps) → target 2 blocks/SM (≤128 regs/thread).
    //                 BM=64  (4 warps) → target 3 blocks/SM (≤170 regs/thread).
    // STAGES_=1: single-buffered (half smem, no load/compute overlap, higher occupancy).
    // STAGES_=2: double-buffered (overlaps decode(next) with compute(current)).
    /**
     * @brief Execute one all-format BK64 tile with an explicit staging schedule.
     * @tparam CODEBOOK_ID Registered physical weight encoding.
     * @tparam BM Number of activation rows owned by the CTA.
     * @tparam BN Number of output columns owned by the CTA.
     * @tparam WARPS_M Warp partition along activation rows.
     * @tparam WARPS_N Warp partition along output columns.
     * @tparam STAGES_ Number of shared operand slots (one or two).
     * @tparam CANONICAL_KPART Whether this CTA publishes a private serial-K slice.
     * @tparam BLOCK_SIZE_ Compile-time CTA thread count.
     * @tparam MIN_BLOCKS_HINT Reviewed register/occupancy launch bound.
     * @tparam STAGING Compile-time operand publication policy; never arithmetic.
     * @param A Persistent quantized activations, row-major [M,K].
     * @param payload Native packed weights, block-major [K/32,N,payload_bytes].
     * @param scales_B Native per-block weight scales.
     * @param mins_B Per-block offset or second scale when required by the format.
     * @param emins_B Packed subgroup offsets for dual-scale asymmetric formats.
     * @param C Disjoint output or persistent canonical-partition workspace.
     * @param scales_A Row-major activation scales [M,K/32].
     * @param sums_A Optional block-major activation sums [K/32,M].
     * @param C_existing Optional existing output for the beta epilogue.
     * @param bias Optional persistent output-column bias.
     * @param M Logical activation rows.
     * @param N Logical output columns.
     * @param K Positive reduction width divisible by 32.
     * @param alpha Partition scale, applied in the canonical arithmetic position.
     * @param beta Existing-output scale, applied only in the final epilogue.
     * @param serial_partitions Immutable public-M1 partition count and span.
     * @param serial_m1_uses_ordered_reducer Whether direct output must retain that fold.
     */
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WARPS_M, int WARPS_N,
              int STAGES_ = 2,
              bool CANONICAL_KPART = false,
              int BLOCK_SIZE_ = WARPS_M * WARPS_N * 32,
              int MIN_BLOCKS_HINT = (STAGES_ == 1)
                                        ? ((BLOCK_SIZE_ >= 256) ? 3 : 4)
                                        : ((BLOCK_SIZE_ >= 256) ? 2 : 3),
              PrefillStagingSchedule STAGING = PrefillStagingSchedule::RegisterDecode>
    __global__ __launch_bounds__(BLOCK_SIZE_, MIN_BLOCKS_HINT) void nativeVnniTC_BK64(
        const int8_t *__restrict__ A,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ scales_B,
        const uint16_t *__restrict__ mins_B,
        const uint32_t *__restrict__ emins_B,
        float *__restrict__ C,
        const float *__restrict__ scales_A,
        const int32_t *__restrict__ sums_A,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        CanonicalM1PartitionGeometry serial_partitions,
        int serial_m1_uses_ordered_reducer)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int ASYNC_RAW_MODE = static_cast<int>(STAGING);
        static_assert(ASYNC_RAW_MODE >= 0 && ASYNC_RAW_MODE <= 3,
                      "unknown NativeVNNI prefill staging schedule");
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16;
        constexpr int WN = WARP_N / 8;
        constexpr int A_VEC_LOADS = BM * BK64 / 16;

        // Named barriers replace __syncthreads() when load/decode are warp-group-affine:
        // - A loads: row-group-affine when A_VEC_LOADS == BLOCK_SIZE (1 cp.async/thread)
        // - B decode: col-group-affine when BLOCK_SIZE >= 2*BN (split-thread path)
        // Each warp hits a row barrier (syncs WARPS_N warps) + col barrier (syncs WARPS_M warps)
        // instead of one full-CTA barrier, reducing barrier stall from 16→4 warp convergence.
        // PERF NOTE: benchmarking showed ~1% regression due to 2× barrier instruction overhead
        // outweighing the smaller sync group benefit. Disabled pending a single-barrier solution.
        [[maybe_unused]] constexpr bool USE_NAMED_BARRIERS = false; // (A_VEC_LOADS == BLOCK_SIZE) && (BLOCK_SIZE >= 2 * BN);

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0);
        static_assert(WARP_N % 8 == 0);
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;
        const int gid = lane_id >> 2;

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;

        const int num_q40_blocks = K / 32;
        const int num_k_tiles_total = (num_q40_blocks + 1) / 2; // ceil: handles K%64!=0
        int canonical_block_begin = 0;
        int canonical_block_end = num_q40_blocks;
        int kt_begin = 0;
        int kt_end = num_k_tiles_total;
        int num_k_iters = num_k_tiles_total;
        if constexpr (CANONICAL_KPART)
        {
            if (serial_partitions.count <= 1)
                return;
            const int blocks_per_partition =
                (num_q40_blocks + serial_partitions.count - 1) /
                serial_partitions.count;
            canonical_block_begin =
                static_cast<int>(blockIdx.z) * blocks_per_partition;
            canonical_block_end = min(
                num_q40_blocks,
                canonical_block_begin + blocks_per_partition);
            kt_begin = canonical_block_begin / 2;
            kt_end = (canonical_block_end + 1) / 2;
            num_k_iters = kt_end - kt_begin;
        }

        // Compile-time traits for this codebook
        using Traits = llaminar2::cuda_native_vnni::CodebookTraits<CODEBOOK_ID>;
        constexpr bool IS_ASYMMETRIC = Traits::is_asymmetric;
        constexpr bool IS_DUAL_SCALE = Traits::is_dual_scale;
        constexpr bool IS_DUAL_SCALE_ASYM = Traits::is_dual_scale_asym;
        constexpr bool IS_IQ1_M = Traits::is_iq1_m;
        constexpr bool NEEDS_MINS = IS_ASYMMETRIC || IS_DUAL_SCALE;

        // Shared memory: no smem_B_raw staging buffer needed.
        // STAGES_=1 (single-buffered) halves smem → potential 3 blocks/SM.
        // STAGES_=2 (double-buffered) overlaps load/decode of next tile with compute.
        __shared__ int8_t smem_A[STAGES_][BM * SMEM_STRIDE_64];
        __shared__ int8_t smem_B[STAGES_][BN * SMEM_STRIDE_64];
        __shared__ uint16_t smem_scales_B[STAGES_][2 * BN];

        // Activation scales cached in smem to decouple from L1 data cache.
        // Stores 2 FP32 scale values per M-row per K-tile (one per Q4_0 block half).
        __shared__ float smem_sa[STAGES_][BM * 2];

        // Asymmetric formats need per-block mins; dual-scale formats need per-block scale_hi.
        // Both use the same smem_mins_B buffer (mins_B pointer carries scale_hi for dual-scale).
        [[maybe_unused]] __shared__ uint16_t smem_mins_B[STAGES_][NEEDS_MINS ? 2 * BN : 1];

        // Q2_K (dual_scale_asym) needs per-block emins: packed {min_lo_fp16, min_hi_fp16} as uint32_t
        [[maybe_unused]] __shared__ uint32_t smem_emins_B[STAGES_][IS_DUAL_SCALE_ASYM ? 2 * BN : 1];

        // IQ1_M needs per-block delta sign bytes from payload (qh0, qh1)
        [[maybe_unused]] __shared__ uint16_t smem_iq1m_qh[STAGES_][IS_IQ1_M ? 2 * BN : 1];

        float acc[WM][WN][4];
        float serial_acc[WM][WN][4];

        // The launcher resolves the immutable span from the canonical serial
        // schedule before capture. Keeping it in the parameter bank avoids
        // both device integer division and an extra long-lived computed
        // register. Only the moving boundary needs a thread-local register.
        int next_serial_partition_end = serial_partitions.blocks_per_partition;

        /**
         * Fold a completed K partition with the same ascending FP32 reducer
         * used by public M=1 KPAR decode.
         *
         * The tensor-core CTA still owns all K work for several rows, so this
         * adds no global partial buffer or extra launch. It merely preserves
         * the serial policy's parenthesization in private registers.
         */
        auto commit_serial_m1_partition =
            [&](int completed_block) __attribute__((always_inline))
        {
            if constexpr (CANONICAL_KPART)
                return;
            if (!serial_m1_uses_ordered_reducer)
                return;

            bool partition_complete;
            if constexpr (IS_IQ1_M)
            {
                // Four subgroup corrections make IQ1_M register-constrained.
                // A live cursor spills its two-CTA tile; reducing occupancy to
                // avoid that spill costs more than the boundary division. Keep
                // the same stateless boundary test for this format, including
                // its original register allocation and concurrent CTA budget.
                const int blocks_per_partition =
                    (num_q40_blocks + serial_partitions.count - 1) /
                    serial_partitions.count;
                partition_complete = completed_block == num_q40_blocks ||
                    (completed_block % blocks_per_partition) == 0;
            }
            else
            {
                partition_complete = completed_block == num_q40_blocks ||
                    completed_block == next_serial_partition_end;
            }
            if (!partition_complete)
                return;

            // A short final partition still commits at num_q40_blocks. Empty
            // trailing partitions require no work here, just as in the
            // previous ascending serial fold; no arithmetic is reassociated.
            if constexpr (!IS_IQ1_M)
                next_serial_partition_end += serial_partitions.blocks_per_partition;

#pragma unroll
            for (int i = 0; i < WM; ++i)
#pragma unroll
                for (int j = 0; j < WN; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        /*
                         * Serial KPAR scales every partition before reducing
                         * it. Applying alpha after the fold is observably
                         * different for non-unit alpha.
                         */
                        const float partial =
                            __fmul_rn(alpha, acc[i][j][e]);
                        serial_acc[i][j][e] = __fadd_rn(
                            serial_acc[i][j][e],
                            partial);
                        acc[i][j][e] = 0.0f;
                    }
        };

        // Load A tile: BM × BK64 bytes via 16-byte async copies
        auto load_A_tile = [&](int stage, int kt) __attribute__((always_inline))
        {
#pragma unroll 2
            for (int idx = threadIdx.x; idx < A_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4;
                const int row = linear >> 6;
                const int col = linear & 63;
                const int grow = block_m + row;
                void *dst = &smem_A[stage][row * SMEM_STRIDE_64 + col];
                const bool valid = grow < M && (kt * BK64 + col + 16 <= K);
                const void *src = valid
                                      ? static_cast<const void *>(&A[static_cast<size_t>(grow) * K + kt * BK64 + col])
                                      : static_cast<const void *>(A);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };


        static_assert(supportsPrefillStaging(
                          STAGING, Traits::payload_bytes, STAGES_, BN, BLOCK_SIZE),
                      "async packed staging requires split-owner aligned nibble payloads");

        /** @brief The metadata view changes ownership only, never its half bits. */
        auto metadata_index = [](int col, int half) {
            return ASYNC_RAW_MODE >= 2 ? half * BN + col : 2 * col + half;
        };

        /**
         * @brief Stage each owner's compressed bytes into its own decoded slot.
         *
         * A copy and the subsequent in-place decode have the same thread owner.
         * No other thread reads this slot until the existing consumer barrier.
         * Four-byte Q5 copies respect its 20-byte stride; Q4 uses one vector.
         */
        auto load_B_raw_tile = [&](int stage, int kt) __attribute__((always_inline))
        {
            if constexpr (ASYNC_RAW_MODE != 0)
            {
                const int col = threadIdx.x & (BN - 1);
                const int half = threadIdx.x / BN;
                if (half >= 2) return;
                const int kb = kt * 2 + half;
                const int gcol = block_n + col;
                const bool valid = kb < num_q40_blocks && gcol < N;
                constexpr int bytes = Traits::payload_bytes;
                constexpr int copy_bytes = bytes == 16 ? 16 : 4;
                const uint8_t *source = valid
                    ? payload + (static_cast<size_t>(kb) * N + gcol) * bytes : payload;
                uint8_t *destination = reinterpret_cast<uint8_t *>(
                    &smem_B[stage][col * SMEM_STRIDE_64 + half * 32]);
#pragma unroll
                for (int offset = 0; offset < bytes; offset += copy_bytes)
                {
                    const uint32_t address = static_cast<uint32_t>(
                        __cvta_generic_to_shared(destination + offset));
                    if constexpr (copy_bytes == 16)
                        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                            :: "r"(address), "l"(source + offset), "r"(valid ? 16 : 0) : "memory");
                    else
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                            :: "r"(address), "l"(source + offset), "r"(valid ? 4 : 0) : "memory");
                }
            }
        };

        /**
         * @brief Publish half metadata in the existing allocation before use.
         *
         * Aligned eight-column groups use native async copies. Ragged source
         * rows require scalar half accesses to preserve alignment and bounds.
         * Compute consumes this view only after every producer has waited and
         * the ordinary CTA barrier has joined the complete stage.
         */
        auto load_B_metadata_tile = [&](int stage, int kt) __attribute__((always_inline))
        {
            if constexpr (ASYNC_RAW_MODE >= 2)
            {
                const int chunk = threadIdx.x;
                if (chunk >= 2 * BN / 8) return;
                const int half = chunk / (BN / 8);
                const int col = (chunk % (BN / 8)) * 8;
                const int kb = kt * 2 + half;
                const int gcol = block_n + col;
                const size_t source_index = static_cast<size_t>(kb) * N + gcol;
                const bool vector_valid = kb < num_q40_blocks && gcol + 8 <= N &&
                                          (source_index & 7) == 0;
                auto stage_metadata = [&](const uint16_t *source, uint16_t *destination) {
                    if (vector_valid)
                        cp_async_cg_16_zfill_128(destination, source + source_index, 16);
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 8; ++i)
                            destination[i] = kb < num_q40_blocks && gcol + i < N
                                ? source[source_index + i] : uint16_t{0};
                    }
                };
                stage_metadata(scales_B, &smem_scales_B[stage][metadata_index(col, half)]);
                if constexpr (NEEDS_MINS)
                    stage_metadata(mins_B, &smem_mins_B[stage][metadata_index(col, half)]);
            }
        };

        // Decode B: load payloads from global memory, decode to int8, write to smem.
        // Split-thread strategy: when BLOCK_SIZE >= 2*BN, first BN threads handle
        // K-block 0 and next BN threads handle K-block 1 in parallel. This doubles
        // warp-level parallelism during decode (100% vs 50% thread utilization).
        // Column-owner schedule for smaller thread counts.
        auto decode_B_direct = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
            const bool has_block1 = (kb0 + 1 < num_q40_blocks);

            if constexpr (BLOCK_SIZE >= 2 * BN)
            {
                // Split: threads [0, BN) → block 0; threads [BN, 2*BN) → block 1.
                // All warps participate, halving per-thread work.
                const int col = threadIdx.x & (BN - 1);  // BN is power of 2
                const int block_half = threadIdx.x / BN; // 0 or 1

                if (block_half >= 2)
                    return; // guard for BLOCK_SIZE > 2*BN

                const int gcol = block_n + col;
                int32_t *dst_words = reinterpret_cast<int32_t *>(&smem_B[stage][col * SMEM_STRIDE_64]);
                const int word_offset = block_half * 8;
                const int scale_slot = block_half;
                const int kb = kb0 + block_half;
                const bool has_this_block = (block_half == 0) || has_block1;

                if (gcol < N && has_this_block)
                {
                    constexpr int PAYLOAD_BYTES = Traits::payload_bytes;
                    const size_t base = (static_cast<size_t>(kb) * N + gcol) * PAYLOAD_BYTES;
                    const uint16_t s = ASYNC_RAW_MODE >= 2 ? uint16_t{0}
                        : scales_B[static_cast<size_t>(kb) * N + gcol];
                    const uint8_t *raw_source = ASYNC_RAW_MODE == 0 ? payload + base
                        : reinterpret_cast<const uint8_t *>(&dst_words[word_offset]);

                    [[maybe_unused]] uint16_t m = 0;
                    if constexpr (NEEDS_MINS && ASYNC_RAW_MODE < 2)
                        m = mins_B[static_cast<size_t>(kb) * N + gcol];

                    [[maybe_unused]] uint32_t em = 0;
                    if constexpr (IS_DUAL_SCALE_ASYM)
                        em = emins_B[static_cast<size_t>(kb) * N + gcol];

                    [[maybe_unused]] uint16_t qh_packed = 0;
                    if constexpr (IS_IQ1_M)
                    {
                        qh_packed = static_cast<uint16_t>(payload[base + 4]) |
                                    (static_cast<uint16_t>(payload[base + 5]) << 8);
                    }

                    // ── Decode one Q-block ──────────────────────────────
                    if constexpr (PAYLOAD_BYTES == 16)
                    {
                        const int4 raw = *reinterpret_cast<const int4 *>(raw_source);
                        const uint32_t r0 = static_cast<uint32_t>(raw.x);
                        const uint32_t r1 = static_cast<uint32_t>(raw.y);
                        const uint32_t r2 = static_cast<uint32_t>(raw.z);
                        const uint32_t r3 = static_cast<uint32_t>(raw.w);
                        if constexpr (CODEBOOK_ID == 0)
                        {
                            *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                                static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                            *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                                static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));
                        }
                        else if constexpr (CODEBOOK_ID == 4)
                        {
                            using llaminar2::cuda_native_vnni::iq4nl_decode_word;
                            uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                            iq4nl_decode_word(r0, lo0, hi0);
                            iq4nl_decode_word(r1, lo1, hi1);
                            iq4nl_decode_word(r2, lo2, hi2);
                            iq4nl_decode_word(r3, lo3, hi3);
                            *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                                static_cast<int32_t>(lo0), static_cast<int32_t>(lo1),
                                static_cast<int32_t>(lo2), static_cast<int32_t>(lo3));
                            *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                                static_cast<int32_t>(hi0), static_cast<int32_t>(hi1),
                                static_cast<int32_t>(hi2), static_cast<int32_t>(hi3));
                        }
                        else
                        {
                            int32_t groups[8];
                            llaminar2::cuda_native_vnni::decode_groups_vec<CODEBOOK_ID>(
                                raw_source, groups);
                            *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                                groups[0], groups[1], groups[2], groups[3]);
                            *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                                groups[4], groups[5], groups[6], groups[7]);
                        }
                    }
                    else
                    {
                        int32_t groups[8];
                        llaminar2::cuda_native_vnni::decode_groups<CODEBOOK_ID>(
                            raw_source, groups);
                        *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(
                            groups[0], groups[1], groups[2], groups[3]);
                        *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(
                            groups[4], groups[5], groups[6], groups[7]);
                    }

                    if constexpr (ASYNC_RAW_MODE < 2)
                        smem_scales_B[stage][metadata_index(col, scale_slot)] = s;
                    if constexpr (NEEDS_MINS && ASYNC_RAW_MODE < 2)
                        smem_mins_B[stage][metadata_index(col, scale_slot)] = m;
                    if constexpr (IS_DUAL_SCALE_ASYM)
                        smem_emins_B[stage][metadata_index(col, scale_slot)] = em;
                    if constexpr (IS_IQ1_M)
                        smem_iq1m_qh[stage][metadata_index(col, scale_slot)] = qh_packed;
                }
                else
                {
                    /*
                     * `ldmatrix` fetches a complete physical fragment even when
                     * only one semantic output column remains. Every invalid
                     * column must therefore be initialized. Leaving gcol>=N
                     * untouched let stale shared-memory bytes participate in
                     * the valid border-column MMA result.
                     */
                    *reinterpret_cast<int4 *>(&dst_words[word_offset]) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4 *>(&dst_words[word_offset + 4]) = make_int4(0, 0, 0, 0);
                    if constexpr (ASYNC_RAW_MODE < 2)
                        smem_scales_B[stage][metadata_index(col, scale_slot)] = uint16_t{0};
                    if constexpr (NEEDS_MINS && ASYNC_RAW_MODE < 2)
                        smem_mins_B[stage][metadata_index(col, scale_slot)] = uint16_t{0};
                    if constexpr (IS_DUAL_SCALE_ASYM)
                        smem_emins_B[stage][metadata_index(col, scale_slot)] = 0u;
                    if constexpr (IS_IQ1_M)
                        smem_iq1m_qh[stage][metadata_index(col, scale_slot)] = 0;
                }
            }
            else
            {
                // Column-owner schedule (BLOCK_SIZE < 2*BN)
                for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
                {
                    const int gcol = block_n + col;
                    int32_t *dst_words = reinterpret_cast<int32_t *>(&smem_B[stage][col * SMEM_STRIDE_64]);

                    if (gcol < N)
                    {
                        constexpr int PAYLOAD_BYTES = Traits::payload_bytes;
                        const size_t base0 = (static_cast<size_t>(kb0) * N + gcol) * PAYLOAD_BYTES;
                        const size_t base1 = has_block1
                                                 ? (static_cast<size_t>(kb0 + 1) * N + gcol) * PAYLOAD_BYTES
                                                 : size_t{0};

                        const uint16_t s0 = scales_B[static_cast<size_t>(kb0) * N + gcol];
                        const uint16_t s1 = has_block1
                                                ? scales_B[static_cast<size_t>(kb0 + 1) * N + gcol]
                                                : uint16_t{0};

                        [[maybe_unused]] uint16_t m0 = 0, m1 = 0;
                        if constexpr (NEEDS_MINS)
                        {
                            m0 = mins_B[static_cast<size_t>(kb0) * N + gcol];
                            if (has_block1)
                                m1 = mins_B[static_cast<size_t>(kb0 + 1) * N + gcol];
                        }

                        [[maybe_unused]] uint32_t em0 = 0, em1 = 0;
                        if constexpr (IS_DUAL_SCALE_ASYM)
                        {
                            em0 = emins_B[static_cast<size_t>(kb0) * N + gcol];
                            if (has_block1)
                                em1 = emins_B[static_cast<size_t>(kb0 + 1) * N + gcol];
                        }

                        [[maybe_unused]] uint16_t qh_packed0 = 0, qh_packed1 = 0;
                        if constexpr (IS_IQ1_M)
                        {
                            qh_packed0 = static_cast<uint16_t>(payload[base0 + 4]) |
                                         (static_cast<uint16_t>(payload[base0 + 5]) << 8);
                            if (has_block1)
                            {
                                qh_packed1 = static_cast<uint16_t>(payload[base1 + 4]) |
                                             (static_cast<uint16_t>(payload[base1 + 5]) << 8);
                            }
                        }

                        // ── Decode block 0 ──────────────────────────────────
                        if constexpr (PAYLOAD_BYTES == 16)
                        {
                            const int4 raw0 = *reinterpret_cast<const int4 *>(payload + base0);
                            const uint32_t r0 = static_cast<uint32_t>(raw0.x);
                            const uint32_t r1 = static_cast<uint32_t>(raw0.y);
                            const uint32_t r2 = static_cast<uint32_t>(raw0.z);
                            const uint32_t r3 = static_cast<uint32_t>(raw0.w);
                            if constexpr (CODEBOOK_ID == 0)
                            {
                                *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                    static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                                *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                    static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                    static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));
                            }
                            else if constexpr (CODEBOOK_ID == 4)
                            {
                                using llaminar2::cuda_native_vnni::iq4nl_decode_word;
                                uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                                iq4nl_decode_word(r0, lo0, hi0);
                                iq4nl_decode_word(r1, lo1, hi1);
                                iq4nl_decode_word(r2, lo2, hi2);
                                iq4nl_decode_word(r3, lo3, hi3);
                                *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                    static_cast<int32_t>(lo0), static_cast<int32_t>(lo1),
                                    static_cast<int32_t>(lo2), static_cast<int32_t>(lo3));
                                *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                    static_cast<int32_t>(hi0), static_cast<int32_t>(hi1),
                                    static_cast<int32_t>(hi2), static_cast<int32_t>(hi3));
                            }
                            else
                            {
                                int32_t groups0[8];
                                llaminar2::cuda_native_vnni::decode_groups_vec<CODEBOOK_ID>(
                                    payload + base0, groups0);
                                *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                    groups0[0], groups0[1], groups0[2], groups0[3]);
                                *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                    groups0[4], groups0[5], groups0[6], groups0[7]);
                            }
                        }
                        else
                        {
                            int32_t groups0[8];
                            llaminar2::cuda_native_vnni::decode_groups<CODEBOOK_ID>(
                                payload + base0, groups0);
                            *reinterpret_cast<int4 *>(&dst_words[0]) = make_int4(
                                groups0[0], groups0[1], groups0[2], groups0[3]);
                            *reinterpret_cast<int4 *>(&dst_words[4]) = make_int4(
                                groups0[4], groups0[5], groups0[6], groups0[7]);
                        }

                        // ── Decode block 1 (K-tail: zero-fill if no second block) ──
                        if (has_block1)
                        {
                            if constexpr (PAYLOAD_BYTES == 16)
                            {
                                const int4 raw1 = *reinterpret_cast<const int4 *>(payload + base1);
                                const uint32_t r0 = static_cast<uint32_t>(raw1.x);
                                const uint32_t r1 = static_cast<uint32_t>(raw1.y);
                                const uint32_t r2 = static_cast<uint32_t>(raw1.z);
                                const uint32_t r3 = static_cast<uint32_t>(raw1.w);
                                if constexpr (CODEBOOK_ID == 0)
                                {
                                    *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                        static_cast<int32_t>(__vsub4(r0 & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4(r1 & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4(r2 & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4(r3 & 0x0F0F0F0Fu, 0x08080808u)));
                                    *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                        static_cast<int32_t>(__vsub4((r0 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4((r1 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4((r2 >> 4) & 0x0F0F0F0Fu, 0x08080808u)),
                                        static_cast<int32_t>(__vsub4((r3 >> 4) & 0x0F0F0F0Fu, 0x08080808u)));
                                }
                                else if constexpr (CODEBOOK_ID == 4)
                                {
                                    using llaminar2::cuda_native_vnni::iq4nl_decode_word;
                                    uint32_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
                                    iq4nl_decode_word(r0, lo0, hi0);
                                    iq4nl_decode_word(r1, lo1, hi1);
                                    iq4nl_decode_word(r2, lo2, hi2);
                                    iq4nl_decode_word(r3, lo3, hi3);
                                    *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                        static_cast<int32_t>(lo0), static_cast<int32_t>(lo1),
                                        static_cast<int32_t>(lo2), static_cast<int32_t>(lo3));
                                    *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                        static_cast<int32_t>(hi0), static_cast<int32_t>(hi1),
                                        static_cast<int32_t>(hi2), static_cast<int32_t>(hi3));
                                }
                                else
                                {
                                    int32_t groups1[8];
                                    llaminar2::cuda_native_vnni::decode_groups_vec<CODEBOOK_ID>(
                                        payload + base1, groups1);
                                    *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                        groups1[0], groups1[1], groups1[2], groups1[3]);
                                    *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                        groups1[4], groups1[5], groups1[6], groups1[7]);
                                }
                            }
                            else
                            {
                                int32_t groups1[8];
                                llaminar2::cuda_native_vnni::decode_groups<CODEBOOK_ID>(
                                    payload + base1, groups1);
                                *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(
                                    groups1[0], groups1[1], groups1[2], groups1[3]);
                                *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(
                                    groups1[4], groups1[5], groups1[6], groups1[7]);
                            }
                        } // has_block1
                        else
                        {
                            *reinterpret_cast<int4 *>(&dst_words[8]) = make_int4(0, 0, 0, 0);
                            *reinterpret_cast<int4 *>(&dst_words[12]) = make_int4(0, 0, 0, 0);
                        }

                        smem_scales_B[stage][metadata_index(col, 0)] = s0;
                        smem_scales_B[stage][metadata_index(col, 1)] = s1;

                        if constexpr (NEEDS_MINS)
                        {
                            smem_mins_B[stage][metadata_index(col, 0)] = m0;
                            smem_mins_B[stage][metadata_index(col, 1)] = m1;
                        }

                        if constexpr (IS_DUAL_SCALE_ASYM)
                        {
                            smem_emins_B[stage][metadata_index(col, 0)] = em0;
                            smem_emins_B[stage][metadata_index(col, 1)] = em1;
                        }

                        if constexpr (IS_IQ1_M)
                        {
                            smem_iq1m_qh[stage][metadata_index(col, 0)] = qh_packed0;
                            smem_iq1m_qh[stage][metadata_index(col, 1)] = qh_packed1;
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int g = 0; g < 16; ++g)
                            dst_words[g] = 0;
                        smem_scales_B[stage][metadata_index(col, 0)] = uint16_t{0};
                        smem_scales_B[stage][metadata_index(col, 1)] = uint16_t{0};
                        if constexpr (NEEDS_MINS)
                        {
                            smem_mins_B[stage][metadata_index(col, 0)] = uint16_t{0};
                            smem_mins_B[stage][metadata_index(col, 1)] = uint16_t{0};
                        }
                        if constexpr (IS_DUAL_SCALE_ASYM)
                        {
                            smem_emins_B[stage][metadata_index(col, 0)] = 0u;
                            smem_emins_B[stage][metadata_index(col, 1)] = 0u;
                        }
                        if constexpr (IS_IQ1_M)
                        {
                            smem_iq1m_qh[stage][metadata_index(col, 0)] = 0;
                            smem_iq1m_qh[stage][metadata_index(col, 1)] = 0;
                        }
                    }
                }
            }
        };

        // Load activation scales for the current K-tile into smem.
        // BM rows × 2 halves (kb0, kb0+1) per K-tile → BM*2 floats.
        auto load_scales_A = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
#pragma unroll 2
            for (int idx = threadIdx.x; idx < BM * 2; idx += BLOCK_SIZE)
            {
                const int row = idx >> 1;
                const int half = idx & 1;
                const int kb = kb0 + half;
                const int grow = block_m + row;
                const bool valid = grow < M && kb < num_q40_blocks;
                if constexpr (ASYNC_RAW_MODE == 3)
                {
                    // The four-byte copy remains aligned even for odd K-block
                    // counts. Its owner commits with payload/weight metadata;
                    // the existing wait and CTA barrier publish the float bits.
                    const float *source = valid
                        ? scales_A + static_cast<size_t>(grow) * num_q40_blocks + kb
                        : scales_A;
                    const uint32_t address = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&smem_sa[stage][idx]));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                        :: "r"(address), "l"(source), "r"(valid ? 4 : 0) : "memory");
                }
                else
                    smem_sa[stage][idx] = valid
                        ? scales_A[grow * num_q40_blocks + kb] : 0.0f;
            }
        };

        const bool is_interior_tile =
            (block_m + BM <= M) && (block_n + BN <= N);

        auto compute_k_tile_interior = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const int kb = kb0 + half;
                if (kb >= num_q40_blocks)
                    break; // K-tail: second q-block doesn't exist
                if constexpr (CANONICAL_KPART)
                {
                    if (kb < canonical_block_begin ||
                        kb >= canonical_block_end)
                    {
                        continue;
                    }
                }
                const int k_offset = half * 32;
                const int scale_slot = half;

                // Pre-load ALL A scales for this half from smem
                float sa_pre[WM][2];
#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int local_row0 = wr * WARP_M + wi * 16 + gid;
                    sa_pre[wi][0] = smem_sa[stage][local_row0 * 2 + half];
                    sa_pre[wi][1] = smem_sa[stage][(local_row0 + 8) * 2 + half];
                }

                /*
                 * Preserve the exact FP16 metadata bits until the shared
                 * decode-equivalent contribution helper consumes them. Earlier
                 * versions expanded these values and then re-expressed every
                 * format's FP32 formula locally, allowing the tensor-core path
                 * to acquire a different contraction tree from serial DP4A.
                 */
                uint16_t scale_bits_pre[WN][2];
                [[maybe_unused]] uint16_t secondary_bits_pre[WN][2];
                [[maybe_unused]] uint32_t emin_bits_pre[WN][2];
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;
                    scale_bits_pre[wj][0] =
                        smem_scales_B[stage][metadata_index(b_col_base + frag_col(lane_id, 0), scale_slot)];
                    scale_bits_pre[wj][1] =
                        smem_scales_B[stage][metadata_index(b_col_base + frag_col(lane_id, 1), scale_slot)];
                    if constexpr (NEEDS_MINS)
                    {
                        secondary_bits_pre[wj][0] =
                            smem_mins_B[stage][metadata_index(b_col_base + frag_col(lane_id, 0), scale_slot)];
                        secondary_bits_pre[wj][1] =
                            smem_mins_B[stage][metadata_index(b_col_base + frag_col(lane_id, 1), scale_slot)];
                    }
                    if constexpr (IS_DUAL_SCALE_ASYM)
                    {
                        emin_bits_pre[wj][0] =
                            smem_emins_B[stage][metadata_index(b_col_base + frag_col(lane_id, 0), scale_slot)];
                        emin_bits_pre[wj][1] =
                            smem_emins_B[stage][metadata_index(b_col_base + frag_col(lane_id, 1), scale_slot)];
                    }
                }

                // ── Pre-load ALL A fragments for this half ────────────
                // Decouples A ldmatrix from MMA, enabling the restructured
                // wj→wi loop that loads each B fragment ONCE instead of WM times.
                uint32_t A_frag_all[WM][4];
                [[maybe_unused]] int sum_A_row0_all[WM], sum_A_row1_all[WM];
                [[maybe_unused]] int sum_A_lo_row0_all[WM], sum_A_lo_row1_all[WM];
                [[maybe_unused]] int sum_A_hi_row0_all[WM], sum_A_hi_row1_all[WM];
                [[maybe_unused]] int sg0_r0_all[WM], sg1_r0_all[WM], sg2_r0_all[WM], sg3_r0_all[WM];
                [[maybe_unused]] int sg0_r1_all[WM], sg1_r1_all[WM], sg2_r1_all[WM], sg3_r1_all[WM];

#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int a_row_base = wr * WARP_M + wi * 16;
                    load_ldmatrix_a_m16n8k32(
                        A_frag_all[wi],
                        reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE_64 + k_offset]),
                        SMEM_STRIDE_64 / 4, lane_id);

                    if constexpr (IS_ASYMMETRIC)
                    {
                        const int grow0 = block_m + a_row_base + gid;
                        const int grow1 = grow0 + 8;
                        if (sums_A)
                        {
                            // Both quantizers publish block-major sums using
                            // this captured M, not the request's active-row
                            // count. Adjacent MMA rows therefore share sectors.
                            sum_A_row0_all[wi] =
                                sums_A[static_cast<size_t>(kb) * M + grow0];
                            sum_A_row1_all[wi] =
                                sums_A[static_cast<size_t>(kb) * M + grow1];
                        }
                        else
                        {
                            const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t s0 = 0, s1 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                            {
                                s0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], s0);
                                s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                            }
                            sum_A_row0_all[wi] = s0;
                            sum_A_row1_all[wi] = s1;
                        }
                    }

                    if constexpr (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                    {
                        const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                        const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                        int32_t slo0 = 0, slo1 = 0, shi0 = 0, shi1 = 0;
#pragma unroll
                        for (int w = 0; w < 4; ++w)
                        {
                            slo0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], slo0);
                            slo1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], slo1);
                        }
#pragma unroll
                        for (int w = 4; w < 8; ++w)
                        {
                            shi0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], shi0);
                            shi1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], shi1);
                        }
                        sum_A_lo_row0_all[wi] = slo0;
                        sum_A_lo_row1_all[wi] = slo1;
                        sum_A_hi_row0_all[wi] = shi0;
                        sum_A_hi_row1_all[wi] = shi1;
                    }

                    if constexpr (IS_IQ1_M)
                    {
                        const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                        const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                        sg0_r0_all[wi] = sg1_r0_all[wi] = sg2_r0_all[wi] = sg3_r0_all[wi] = 0;
                        sg0_r1_all[wi] = sg1_r1_all[wi] = sg2_r1_all[wi] = sg3_r1_all[wi] = 0;
                        for (int w = 0; w < 2; ++w)
                        {
                            sg0_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg0_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                        for (int w = 2; w < 4; ++w)
                        {
                            sg1_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg1_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                        for (int w = 4; w < 6; ++w)
                        {
                            sg2_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg2_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                        for (int w = 6; w < 8; ++w)
                        {
                            sg3_r0_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row0_ptr)[w]);
                            sg3_r1_all[wi] += llaminar2::cuda_native_vnni::sum_packed_i8(reinterpret_cast<const int32_t *>(row1_ptr)[w]);
                        }
                    }
                }

                // ── Compute: wj (outer) → wi (inner) ─────────────────
                // B fragment loaded ONCE per wj, reused across all WM rows.
                // Halves B ldmatrix loads vs the original wi→wj order.
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    uint32_t B_frag[2];
                    load_ldmatrix_b_m16n8k32(
                        B_frag,
                        reinterpret_cast<const int *>(&smem_B[stage][(wc * WARP_N + wj * 8) * SMEM_STRIDE_64 + k_offset]),
                        SMEM_STRIDE_64 / 4, lane_id);

#pragma unroll
                    for (int wi = 0; wi < WM; ++wi)
                    {
                        const float sa0 = sa_pre[wi][0];
                        const float sa1 = sa_pre[wi][1];

                        int32_t dot_lo[4] = {0, 0, 0, 0};
                        int32_t dot_hi[4] = {0, 0, 0, 0};
                        if constexpr (IS_DUAL_SCALE)
                        {
                            const uint32_t B_lo[2] = {B_frag[0], 0u};
                            const uint32_t B_hi[2] = {0u, B_frag[1]};
                            mma_m16n8k32_s8(
                                dot_lo,
                                A_frag_all[wi],
                                B_lo);
                            mma_m16n8k32_s8(
                                dot_hi,
                                A_frag_all[wi],
                                B_hi);
                        }
                        else
                        {
                            mma_m16n8k32_s8(
                                dot_lo,
                                A_frag_all[wi],
                                B_frag);
                        }

                        const int b_col_base =
                            wc * WARP_N + wj * 8;
                        const uint16_t qh_c0 = IS_IQ1_M
                            ? smem_iq1m_qh[stage][
                                  2 * (b_col_base +
                                       frag_col(lane_id, 0)) +
                                  scale_slot]
                            : uint16_t{0};
                        const uint16_t qh_c1 = IS_IQ1_M
                            ? smem_iq1m_qh[stage][
                                  2 * (b_col_base +
                                       frag_col(lane_id, 1)) +
                                  scale_slot]
                            : uint16_t{0};

#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            const int column = e & 1;
                            const bool first_row = e < 2;
                            const float activation_scale =
                                first_row ? sa0 : sa1;
                            const int activation_sum =
                                IS_ASYMMETRIC
                                    ? (first_row
                                           ? sum_A_row0_all[wi]
                                           : sum_A_row1_all[wi])
                                    : 0;
                            const int activation_sum_lo =
                                (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                    ? (first_row
                                           ? sum_A_lo_row0_all[wi]
                                           : sum_A_lo_row1_all[wi])
                                    : 0;
                            const int activation_sum_hi =
                                (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                    ? (first_row
                                           ? sum_A_hi_row0_all[wi]
                                           : sum_A_hi_row1_all[wi])
                                    : 0;
                            const uint32_t emin_bits =
                                IS_DUAL_SCALE_ASYM
                                    ? emin_bits_pre[wj][column]
                                    : uint32_t{0};
                            const uint16_t secondary_bits =
                                NEEDS_MINS
                                    ? secondary_bits_pre[wj][column]
                                    : uint16_t{0};
                            const uint16_t iq1m_qh =
                                column ? qh_c1 : qh_c0;
                            const int subgroup_sum0 = IS_IQ1_M
                                ? (first_row
                                       ? sg0_r0_all[wi]
                                       : sg0_r1_all[wi])
                                : 0;
                            const int subgroup_sum1 = IS_IQ1_M
                                ? (first_row
                                       ? sg1_r0_all[wi]
                                       : sg1_r1_all[wi])
                                : 0;
                            const int subgroup_sum2 = IS_IQ1_M
                                ? (first_row
                                       ? sg2_r0_all[wi]
                                       : sg2_r1_all[wi])
                                : 0;
                            const int subgroup_sum3 = IS_IQ1_M
                                ? (first_row
                                       ? sg3_r0_all[wi]
                                       : sg3_r1_all[wi])
                                : 0;
                            const float contribution =
                                llaminar2::cuda_native_vnni::
                                    native_vnni_block_contribution_from_reduced_terms_rn<
                                        CODEBOOK_ID>(
                                        dot_lo[e],
                                        dot_hi[e],
                                        activation_scale,
                                        scale_bits_pre[wj][column],
                                        secondary_bits,
                                        emin_bits,
                                        activation_sum,
                                        activation_sum_lo,
                                        activation_sum_hi,
                                        iq1m_qh,
                                        subgroup_sum0,
                                        subgroup_sum1,
                                        subgroup_sum2,
                                        subgroup_sum3);
                            acc[wi][wj][e] = __fadd_rn(
                                acc[wi][wj][e],
                                contribution);
                        }
                    }
                }
                commit_serial_m1_partition(kb + 1);
            }
        };

        auto compute_k_tile_border = [&](int stage, int kt) __attribute__((always_inline))
        {
            const int kb0 = kt * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const int kb = kb0 + half;
                if (kb >= num_q40_blocks)
                    break; // K-tail: second q-block doesn't exist
                if constexpr (CANONICAL_KPART)
                {
                    if (kb < canonical_block_begin ||
                        kb >= canonical_block_end)
                    {
                        continue;
                    }
                }
                const int k_offset = half * 32;
                const int scale_slot = half;

#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int a_row_base = wr * WARP_M + wi * 16;
                    uint32_t A_frag[4];
                    load_ldmatrix_a_m16n8k32(
                        A_frag,
                        reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE_64 + k_offset]),
                        SMEM_STRIDE_64 / 4, lane_id);

                    const int grow0 = block_m + a_row_base + gid;
                    const int grow1 = grow0 + 8;
                    const int local_row0 = a_row_base + gid;
                    const float sa0 = (grow0 < M) ? smem_sa[stage][local_row0 * 2 + half] : 0.0f;
                    const float sa1 = (grow1 < M) ? smem_sa[stage][(local_row0 + 8) * 2 + half] : 0.0f;

                    // sum_A for asymmetric correction (border variant with bounds check)
                    [[maybe_unused]] int sum_A_row0 = 0, sum_A_row1 = 0;
                    if constexpr (IS_ASYMMETRIC)
                    {
                        if (sums_A)
                        {
                            if (grow0 < M)
                                sum_A_row0 =
                                    sums_A[static_cast<size_t>(kb) * M + grow0];
                            if (grow1 < M)
                                sum_A_row1 =
                                    sums_A[static_cast<size_t>(kb) * M + grow1];
                        }
                        else if (grow0 < M)
                        {
                            const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                            int32_t s0 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                                s0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], s0);
                            sum_A_row0 = s0;
                            if (grow1 < M)
                            {
                                const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                                int32_t s1 = 0;
#pragma unroll
                                for (int w = 0; w < 8; ++w)
                                    s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                                sum_A_row1 = s1;
                            }
                        }
                        else if (grow1 < M)
                        {
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t s1 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                                s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                            sum_A_row1 = s1;
                        }
                    }

                    // Split sums for dual_scale_asym (Q2_K) and IQ1_M
                    [[maybe_unused]] int sum_A_lo_row0 = 0, sum_A_lo_row1 = 0;
                    [[maybe_unused]] int sum_A_hi_row0 = 0, sum_A_hi_row1 = 0;
                    if constexpr (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                    {
                        if (grow0 < M)
                        {
                            const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                            int32_t slo = 0, shi = 0;
                            for (int w = 0; w < 4; ++w)
                                slo = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], slo);
                            for (int w = 4; w < 8; ++w)
                                shi = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], shi);
                            sum_A_lo_row0 = slo;
                            sum_A_hi_row0 = shi;
                        }
                        if (grow1 < M)
                        {
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t slo = 0, shi = 0;
                            for (int w = 0; w < 4; ++w)
                                slo = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], slo);
                            for (int w = 4; w < 8; ++w)
                                shi = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], shi);
                            sum_A_lo_row1 = slo;
                            sum_A_hi_row1 = shi;
                        }
                    }

                    [[maybe_unused]] int subgroup_r0[4] = {0, 0, 0, 0};
                    [[maybe_unused]] int subgroup_r1[4] = {0, 0, 0, 0};
                    if constexpr (IS_IQ1_M)
                    {
                        if (grow0 < M)
                        {
                            const int32_t *row0_words =
                                reinterpret_cast<const int32_t *>(
                                    &smem_A[stage][
                                        (a_row_base + gid) *
                                            SMEM_STRIDE_64 +
                                        k_offset]);
#pragma unroll
                            for (int word = 0; word < 8; ++word)
                            {
                                subgroup_r0[word / 2] +=
                                    llaminar2::cuda_native_vnni::
                                        sum_packed_i8(row0_words[word]);
                            }
                        }
                        if (grow1 < M)
                        {
                            const int32_t *row1_words =
                                reinterpret_cast<const int32_t *>(
                                    &smem_A[stage][
                                        (a_row_base + gid + 8) *
                                            SMEM_STRIDE_64 +
                                        k_offset]);
#pragma unroll
                            for (int word = 0; word < 8; ++word)
                            {
                                subgroup_r1[word / 2] +=
                                    llaminar2::cuda_native_vnni::
                                        sum_packed_i8(row1_words[word]);
                            }
                        }
                    }

#pragma unroll
                    for (int wj = 0; wj < WN; ++wj)
                    {
                        const int b_col_base = wc * WARP_N + wj * 8;
                        const int lc0 = b_col_base + frag_col(lane_id, 0);
                        const int lc1 = b_col_base + frag_col(lane_id, 1);
                        const bool column0_valid = block_n + lc0 < N;
                        const bool column1_valid = block_n + lc1 < N;
                        const uint16_t scale_bits[2] = {
                            column0_valid
                                ? smem_scales_B[stage][metadata_index(lc0, scale_slot)]
                                : uint16_t{0},
                            column1_valid
                                ? smem_scales_B[stage][metadata_index(lc1, scale_slot)]
                                : uint16_t{0}};
                        const uint16_t secondary_bits[2] = {
                            NEEDS_MINS && column0_valid
                                ? smem_mins_B[stage][metadata_index(lc0, scale_slot)]
                                : uint16_t{0},
                            NEEDS_MINS && column1_valid
                                ? smem_mins_B[stage][metadata_index(lc1, scale_slot)]
                                : uint16_t{0}};
                        const uint32_t emin_bits[2] = {
                            IS_DUAL_SCALE_ASYM && column0_valid
                                ? smem_emins_B[stage][metadata_index(lc0, scale_slot)]
                                : uint32_t{0},
                            IS_DUAL_SCALE_ASYM && column1_valid
                                ? smem_emins_B[stage][metadata_index(lc1, scale_slot)]
                                : uint32_t{0}};
                        const uint16_t iq1m_qh[2] = {
                            IS_IQ1_M && column0_valid
                                ? smem_iq1m_qh[stage][metadata_index(lc0, scale_slot)]
                                : uint16_t{0},
                            IS_IQ1_M && column1_valid
                                ? smem_iq1m_qh[stage][metadata_index(lc1, scale_slot)]
                                : uint16_t{0}};

                        uint32_t B_frag[2];
                        load_ldmatrix_b_m16n8k32(
                            B_frag,
                            reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE_64 + k_offset]),
                            SMEM_STRIDE_64 / 4, lane_id);

                        int32_t dot_lo[4] = {0, 0, 0, 0};
                        int32_t dot_hi[4] = {0, 0, 0, 0};
                        if constexpr (IS_DUAL_SCALE)
                        {
                            const uint32_t B_lo[2] = {B_frag[0], 0u};
                            const uint32_t B_hi[2] = {0u, B_frag[1]};
                            mma_m16n8k32_s8(dot_lo, A_frag, B_lo);
                            mma_m16n8k32_s8(dot_hi, A_frag, B_hi);
                        }
                        else
                        {
                            mma_m16n8k32_s8(dot_lo, A_frag, B_frag);
                        }

#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            const int column = e & 1;
                            const bool first_row = e < 2;
                            const float activation_scale =
                                first_row ? sa0 : sa1;
                            const int *subgroup_sums =
                                first_row ? subgroup_r0 : subgroup_r1;
                            const float contribution =
                                llaminar2::cuda_native_vnni::
                                    native_vnni_block_contribution_from_reduced_terms_rn<
                                        CODEBOOK_ID>(
                                        dot_lo[e],
                                        dot_hi[e],
                                        activation_scale,
                                        scale_bits[column],
                                        secondary_bits[column],
                                        emin_bits[column],
                                        IS_ASYMMETRIC
                                            ? (first_row
                                                   ? sum_A_row0
                                                   : sum_A_row1)
                                            : 0,
                                        (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                            ? (first_row
                                                   ? sum_A_lo_row0
                                                   : sum_A_lo_row1)
                                            : 0,
                                        (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                            ? (first_row
                                                   ? sum_A_hi_row0
                                                   : sum_A_hi_row1)
                                            : 0,
                                        iq1m_qh[column],
                                        IS_IQ1_M ? subgroup_sums[0] : 0,
                                        IS_IQ1_M ? subgroup_sums[1] : 0,
                                        IS_IQ1_M ? subgroup_sums[2] : 0,
                                        IS_IQ1_M ? subgroup_sums[3] : 0);
                            acc[wi][wj][e] = __fadd_rn(
                                acc[wi][wj][e],
                                contribution);
                        }
                    }
                }
                commit_serial_m1_partition(kb + 1);
            }
        };

        // One output tile follows either the full-K walk or the exact
        // public-M1 K-partition boundaries.
        {
            // No lambda wrapping — keeps variables const and avoids capture overhead.

#pragma unroll
            for (int i = 0; i < WM; ++i)
#pragma unroll
                for (int j = 0; j < WN; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        acc[i][j][e] = 0.0f;
                        serial_acc[i][j][e] = 0.0f;
                    }

            if (num_k_iters > 0)
            {
                if constexpr (STAGES_ == 1)
                {
                    // ── Single-buffered main loop ────────────────────────
                    for (int ki = 0; ki < num_k_iters; ++ki)
                    {
                        const int kt = kt_begin + ki;

                        load_A_tile(0, kt);
                        cp_async_commit();
                        decode_B_direct(0, kt);
                        load_scales_A(0, kt);
                        cp_async_wait<0>();
                        __syncthreads();

                        if (is_interior_tile)
                            compute_k_tile_interior(0, kt);
                        else
                            compute_k_tile_border(0, kt);

                        if (ki + 1 < num_k_iters)
                            __syncthreads();
                    }
                }
                else if constexpr (ASYNC_RAW_MODE != 0)
                {
                    // Prime compressed payload and activations together. Each
                    // decoder waits for its own raw slot; the CTA then joins
                    // decoded bytes and metadata before any ldmatrix consumer.
                    load_A_tile(0, kt_begin);
                    load_B_raw_tile(0, kt_begin);
                    load_B_metadata_tile(0, kt_begin);
                    if constexpr (ASYNC_RAW_MODE == 3)
                        load_scales_A(0, kt_begin);
                    cp_async_commit();
                    if constexpr (ASYNC_RAW_MODE != 3)
                        load_scales_A(0, kt_begin);
                    cp_async_wait<0>();
                    decode_B_direct(0, kt_begin);
                    __syncthreads();
                    for (int ki = 0; ki < num_k_iters; ++ki)
                    {
                        const int stage = ki & 1;
                        const int kt = kt_begin + ki;
                        if (ki + 1 < num_k_iters)
                        {
                            load_A_tile(stage ^ 1, kt + 1);
                            load_B_raw_tile(stage ^ 1, kt + 1);
                            load_B_metadata_tile(stage ^ 1, kt + 1);
                            if constexpr (ASYNC_RAW_MODE == 3)
                                load_scales_A(stage ^ 1, kt + 1);
                            cp_async_commit();
                            if constexpr (ASYNC_RAW_MODE != 3)
                                load_scales_A(stage ^ 1, kt + 1);
                        }
                        if (is_interior_tile)
                            compute_k_tile_interior(stage, kt);
                        else
                            compute_k_tile_border(stage, kt);
                        if (ki + 1 < num_k_iters)
                        {
                            cp_async_wait<0>();
                            decode_B_direct(stage ^ 1, kt + 1);
                            // The opposite slot cannot be overwritten by the
                            // next iteration until every old consumer is done.
                            __syncthreads();
                        }
                    }
                }
                else
                {
                    // ── Double-buffered pipeline (STAGES_=2) ─────────────
                    load_A_tile(0, kt_begin);
                    cp_async_commit();
                    decode_B_direct(0, kt_begin);
                    load_scales_A(0, kt_begin);
                    cp_async_wait<0>();
                    __syncthreads();

                    for (int ki = 0; ki < num_k_iters; ++ki)
                    {
                        const int stage = ki & 1;
                        const int kt = kt_begin + ki;

                        if (ki + 1 < num_k_iters)
                        {
                            load_A_tile(stage ^ 1, kt + 1);
                            cp_async_commit();
                            decode_B_direct(stage ^ 1, kt + 1);
                            load_scales_A(stage ^ 1, kt + 1);
                        }

                        if (is_interior_tile)
                            compute_k_tile_interior(stage, kt);
                        else
                            compute_k_tile_border(stage, kt);

                        if (ki + 1 < num_k_iters)
                        {
                            cp_async_wait<0>();
                            __syncthreads();
                        }
                    }
                }
            }

            float output_alpha = alpha;
            if constexpr (!CANONICAL_KPART)
            {
                if (serial_m1_uses_ordered_reducer)
                {
#pragma unroll
                    for (int i = 0; i < WM; ++i)
#pragma unroll
                        for (int j = 0; j < WN; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = serial_acc[i][j][e];
                    output_alpha = 1.0f;
                }
            }

            // Epilogue: write accumulators to global memory
            const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

            // Canonical K-partition CTAs write private z slices. The ordered
            // reducer applies beta and bias after folding those slices.
            float *__restrict__ C_out = C;
            if constexpr (CANONICAL_KPART)
                C_out = C + blockIdx.z * M * N;

#pragma unroll
            for (int wj = 0; wj < WN; ++wj)
            {
                const int tile_n = block_n + wc * WARP_N + wj * 8;
                const int gc0 = tile_n + frag_col(lane_id, 0);
                const int gc1 = tile_n + frag_col(lane_id, 1);
                const bool gc0_valid = gc0 < N;
                const bool gc1_valid = gc1 < N;
                const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
                const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;

#pragma unroll
                for (int wi = 0; wi < WM; ++wi)
                {
                    const int tile_m = block_m + wr * WARP_M + wi * 16;
                    const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);

                    if (interior && (simple_epilogue || CANONICAL_KPART))
                    {
                        const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                        const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                        const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                        const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                        // Canonical partials defer beta and bias to the reducer.
                        C_out[out_idx0] = acc[wi][wj][0] * output_alpha;
                        C_out[out_idx1] = acc[wi][wj][1] * output_alpha;
                        C_out[out_idx2] = acc[wi][wj][2] * output_alpha;
                        C_out[out_idx3] = acc[wi][wj][3] * output_alpha;
                        continue;
                    }

#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const int gr = tile_m + frag_row(lane_id, e);
                        const int gc = (e & 1) ? gc1 : gc0;

                        if (gr < M && gc < N)
                        {
                            const int out_idx = gr * N + gc;
                            float val = acc[wi][wj][e] * output_alpha;

                            if constexpr (!CANONICAL_KPART)
                            {
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                            }
                            C_out[out_idx] = val;
                        }
                    }
                }
            }
        }

#else
        (void)A;
        (void)payload;
        (void)scales_B;
        (void)C;
        (void)scales_A;
        (void)C_existing;
        (void)bias;
        (void)M;
        (void)N;
        (void)K;
        (void)alpha;
        (void)beta;
        (void)serial_partitions;
        (void)serial_m1_uses_ordered_reducer;
#endif
    }

} // namespace llaminar2::cuda::prefill
