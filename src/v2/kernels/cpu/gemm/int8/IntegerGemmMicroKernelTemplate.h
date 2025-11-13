/**
 * @file IntegerGemmMicroKernelTemplate.h
 * @brief Clean INT8 micro-kernel template mirroring FP32 design
 *
 * Design principles (following GemmMicroKernelTemplateFP32.h):
 * 1. Single INT32 accumulator (no dual accumulators)
 * 2. Pure INT32 computation until reduction
 * 3. Scales applied ONCE during reduction (not per K-block)
 * 4. Clean VNNI intrinsics (no bias correction hacks)
 * 5. Simple interface matching FP32 pattern
 *
 * Algorithm:
 * ```
 * zero(): acc[TILE_M][TILE_N] = 0 (INT32)
 *
 * accumulate(): acc += VNNI(A_int8, B_int8)  // Pure INT32, no scale yet
 *
 * reduce():
 *   for (i,j): fp_tile[i][j] = (float)acc[i][j] * row_scale[i] * col_scale[j]
 *   quantize(fp_tile) → Q8_0 blocks
 * ```
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#pragma once

#include "../../SimdTraits.h"
#include "../../../../tensors/FP16Utils.h"
#include "../../../../tensors/BlockStructures.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {
            using ::llaminar2::Q8_0Block;

            /**
             * @brief Integer micro-kernel: INT8×INT8→INT32→Q8_0
             *
             * @tparam ISA Instruction set architecture (AVX512VNNITag for VNNI support)
             * @tparam MR Micro-kernel M dimension (rows in register block)
             * @tparam NR Micro-kernel N dimension (cols in register block, must be 32)
             * @tparam K_BLOCKS_PER_ITER Number of Q8_0 blocks per iteration (1, 2, 4, or 8) - determines scale stride
             * @tparam UNROLL_K K-loop unroll factor (for outer loop control)
             * @tparam PREFETCH_DIST Prefetch distance (for outer loop control)
             * @tparam MC M-dimension cache block size (unused in micro-kernel)
             * @tparam KC K-dimension cache block size (unused in micro-kernel)
             * @tparam NC N-dimension cache block size (unused in micro-kernel)
             *
             * Interface matches FP32 micro-kernel pattern for consistency.
             */
            template <
                typename ISA,
                int MR,
                int NR,
                int K_BLOCKS_PER_ITER = 2,
                int UNROLL_K = 4,
                int PREFETCH_DIST = 2,
                int MC = 256,
                int KC = 512,
                int NC = 128>
            class IntegerMicroKernelTemplate
            {
            public:
                using Traits = simd::SimdTraits<ISA>;

                static constexpr int TILE_M = MR;
                static constexpr int TILE_N = NR;

                // Q8_0 blocks are always 32 elements
                static_assert(TILE_N == 32, "TILE_N must be 32 for Q8_0 block alignment");

                IntegerMicroKernelTemplate()
                {
                    zero();
                }

                /**
                 * @brief Zero all accumulators
                 */
                inline void zero()
                {
                    std::memset(acc_, 0, sizeof(acc_));
                    std::memset(fp_acc_, 0, sizeof(fp_acc_));
                }

                /**
                 * @brief Accumulate A_panel × B_panel into INT32 accumulator
                 *
                 * @param A_panel INT8 A panel [TILE_M × k_panel] (row-major)
                 * @param B_panel INT8 B panel [TILE_N × k_panel] (row-major)
                 * @param k_panel Panel width (32 for single Q8_0 block)
                 * @param a_scales Row scales (FP32, TILE_M elements)
                 * @param b_scales Column scales (FP32, TILE_N elements)
                 *
                 * Note: Accumulates FP32 values with scales applied immediately since each K-block
                 * has its own scales (not a single global scale).
                 */
                inline void accumulate(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    int k_panel,
                    const float *a_scales,
                    const float *b_scales)
                {
#if defined(__AVX512VNNI__)
                    // Ultra-wide path: Process 8 Q8_0 blocks (256 bytes) at once
                    // Uses four 64-byte SIMD registers for maximum memory bandwidth!
                    if (k_panel == 256)
                    {
                        accumulate_vnni_256_with_scales(A_panel, B_panel, a_scales, b_scales);
                    }
                    // Optimal path: Process 4 Q8_0 blocks (128 bytes) at once
                    // Uses two 64-byte SIMD registers for maximum throughput!
                    else if (k_panel == 128)
                    {
                        accumulate_vnni_128_with_scales(A_panel, B_panel, a_scales, b_scales);
                    }
                    // Process 2 Q8_0 blocks (64 bytes) at once
                    // Uses full 64-byte SIMD register and all 16 DPBUSD lanes
                    else if (k_panel == 64)
                    {
                        accumulate_vnni_64_with_scales(A_panel, B_panel, a_scales, b_scales);
                    }
                    // Single block path: Process 1 Q8_0 block (32 bytes)
                    else if (k_panel == 32)
                    {
                        accumulate_vnni_32_with_scales(A_panel, B_panel, a_scales, b_scales);
                    }
                    else
                    {
                        accumulate_scalar_with_scales(A_panel, B_panel, k_panel, a_scales, b_scales);
                    }
#else
                    // Fallback to scalar path if VNNI not available
                    accumulate_scalar_with_scales(A_panel, B_panel, k_panel, a_scales, b_scales);
#endif
                }

                /**
                 * @brief Zero-copy accumulate: Work directly from Q8_0 blocks (NO memcpy!)
                 *
                 * This is the CRITICAL performance optimization - eliminates ~500K memcpy calls.
                 * Works directly from source Q8_0 blocks using pointer arithmetic.
                 *
                 * @param A_blocks Pointer to A matrix Q8_0 blocks [m × k_blocks]
                 * @param B_provider B matrix block provider (row-major access)
                 * @param tile_m Actual M tile size (may be < TILE_M at boundary)
                 * @param tile_n Actual N tile size (must be 32 for Q8_0 alignment)
                 * @param ii Row offset in A matrix
                 * @param jj Column offset in B matrix
                 * @param kb K-block offset
                 * @param k_blocks Total K-blocks in matrices
                 * @param blocks_to_process Number of K-blocks to process (1, 2, 4, or 8)
                 */
                template <typename BlockProvider>
                inline void accumulate_zerocopy(
                    const Q8_0Block *A_blocks,
                    BlockProvider &B_provider,
                    int tile_m,
                    int tile_n,
                    int ii,
                    int jj,
                    size_t kb,
                    size_t k_blocks,
                    int blocks_to_process)
                {
                    const int k_panel = blocks_to_process * 32;

#if defined(__AVX512VNNI__)
                    // Dispatch based on number of blocks (determines SIMD width)
                    if (blocks_to_process == 8)
                    {
                        accumulate_vnni_256_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks);
                    }
                    else if (blocks_to_process == 4)
                    {
                        accumulate_vnni_128_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks);
                    }
                    else if (blocks_to_process == 2)
                    {
                        accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks);
                    }
                    else // blocks_to_process == 1
                    {
                        accumulate_vnni_32_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks);
                    }
#else
                    // Fallback scalar path
                    accumulate_scalar_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks, k_panel);
#endif
                }

                /**
                 * @brief Reduce accumulators to Q8_0 blocks
                 *
                 * Since scales are already applied during accumulation, we just
                 * quantize the FP32 accumulator directly to Q8_0.
                 */
                inline void reduce(Q8_0Block *C_blocks)
                {
                    // Quantize each row: FP32[32] → Q8_0Block
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        quantize_fp32_to_q8_0(&fp_acc_[i * TILE_N], &C_blocks[i]);
                    }
                }

                /**
                 * @brief Get raw INT32 accumulator value (for debugging)
                 */
                inline int32_t get_accumulator(int i, int j) const
                {
                    return acc_[i * TILE_N + j];
                }

            private:
                /**
                 * @brief Scalar accumulation with immediate scale application
                 *
                 * Since each K-block has its own scales, we must apply scales
                 * immediately during accumulation (not defer to reduction phase).
                 */
                inline void accumulate_scalar_with_scales(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    int k_panel,
                    const float *a_scales,
                    const float *b_scales)
                {
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * k_panel;
                        float a_scale = a_scales[i];

                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * k_panel;
                            float b_scale = b_scales[j];

                            int32_t dot = 0;
                            for (int k = 0; k < k_panel; ++k)
                            {
                                dot += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_col[k]);
                            }

                            // Apply scales immediately and accumulate to FP32 (matches llama.cpp)
                            fp_acc_[i * TILE_N + j] += static_cast<float>(dot) * a_scale * b_scale;
                        }
                    }
                }

                /**
                 * @brief Scalar accumulation fallback (old version without scales)
                 */
                inline void accumulate_scalar(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    int k_panel)
                {
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * k_panel;
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * k_panel;

                            int32_t dot = 0;
                            for (int k = 0; k < k_panel; ++k)
                            {
                                dot += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_col[k]);
                            }

                            acc_[i * TILE_N + j] += dot;
                        }
                    }
                }

                /**
                 * @brief Zero-copy scalar accumulation (NO memcpy - works directly from Q8_0 blocks!)
                 *
                 * This eliminates the ~500K memcpy overhead by using pointer arithmetic to access
                 * source Q8_0 blocks directly.
                 */
                template <typename BlockProvider>
                inline void accumulate_scalar_zerocopy(
                    const Q8_0Block *A_blocks,
                    BlockProvider &B_provider,
                    int tile_m,
                    int tile_n,
                    int ii,
                    int jj,
                    size_t kb,
                    size_t k_blocks,
                    int k_panel)
                {
                    const int blocks_to_process = k_panel / 32;

                    for (int i = 0; i < tile_m; ++i)
                    {
                        for (int j = 0; j < tile_n; ++j)
                        {
                            float accumulated_fp = 0.0f;

                            // Process each K-block directly from source
                            for (int b = 0; b < blocks_to_process; ++b)
                            {
                                // Direct pointer to A block (no copy!)
                                const Q8_0Block *a_block = &A_blocks[(ii + i) * k_blocks + kb + b];

                                // Direct pointer to B block from provider (no copy!)
                                const Q8_0Block *b_block = B_provider.get_q8_block(jj + j, kb + b);

                                // Extract scales (FP16 → FP32)
                                float a_scale = fp16_to_fp32(a_block->d);
                                float b_scale = fp16_to_fp32(b_block->d);

                                // Compute dot product directly from source INT8 data
                                int32_t dot = 0;
                                for (int k = 0; k < 32; ++k)
                                {
                                    dot += static_cast<int32_t>(a_block->qs[k]) *
                                           static_cast<int32_t>(b_block->qs[k]);
                                }

                                // Accumulate with scales
                                accumulated_fp += static_cast<float>(dot) * a_scale * b_scale;
                            }

                            fp_acc_[i * TILE_N + j] += accumulated_fp;
                        }
                    }
                }

#if defined(__AVX512VNNI__)
                /**
                 * @brief AVX512-VNNI optimized accumulation for k_panel=32 with immediate scale application
                 *
                 * Uses _mm512_dpbusd_epi32(acc, a, b) which computes: acc += UNSIGNED(a) × SIGNED(b)
                 *
                 * Note: dpbusd interprets A as unsigned [0, 255], but our Q8_0 codes are signed [-127, 127].
                 * Bias correction: dot_signed = dot_unsigned - 256·Σ(B[i]·I(A[i]<0))
                 */
                inline void accumulate_vnni_32_with_scales(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    const float *a_scales,
                    const float *b_scales)
                {
                    // Load mask for 32 elements (rest of 64-byte register zeroed)
                    const __mmask64 mask32 = 0x00000000FFFFFFFFULL;

                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * 32;
                        float a_scale = a_scales[i * 2]; // Stride-2 for consistency with 64-byte path

                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * 32;
                            float b_scale = b_scales[j * 2]; // Stride-2 for consistency

                            // Load 32 INT8 elements (zero upper 32 bytes)
                            __m512i a_vec = _mm512_maskz_loadu_epi8(mask32, a_row);
                            __m512i b_vec = _mm512_maskz_loadu_epi8(mask32, b_col);

                            // VNNI: UNSIGNED(a) × SIGNED(b)
                            __m512i zero = _mm512_setzero_si512();
                            __m512i result = _mm512_dpbusd_epi32(zero, a_vec, b_vec);

                            // Horizontal reduction: sum FIRST 8 INT32 lanes (8 lanes × 4 bytes = 32 bytes)
                            // Note: dpbusd computes 16 separate 4-byte dot products, we only use lanes 0-7
                            // Use AVX512 reduce with mask to sum only valid lanes (SIMD-optimized!)
                            const __mmask16 lane_mask_8 = 0xFF; // Mask for first 8 INT32 lanes (bits 0-7 set)
                            int32_t unsigned_sum = _mm512_mask_reduce_add_epi32(lane_mask_8, result);

                            // Bias correction: Convert unsigned interpretation of A to signed
                            // dpbusd interprets A as unsigned, but we have signed values
                            // For signed value a ∈ [-128, 127], unsigned interpretation is:
                            //   - If a ≥ 0: unsigned = a
                            //   - If a < 0: unsigned = 256 + a  (two's complement wraparound)
                            //
                            // For dot product:
                            // dot_signed = Σ(A_signed[i] * B[i])
                            // dot_unsigned = Σ(A_unsigned[i] * B[i]) = Σ((A_signed[i] + 256·I(A[i]<0)) * B[i])
                            //              = Σ(A_signed[i] * B[i]) + 256·Σ(B[i]·I(A[i]<0))
                            //
                            // Where I(A[i]<0) = 1 if A[i] < 0, else 0
                            //
                            // So: dot_signed = dot_unsigned - 256·Σ(B[i]·I(A[i]<0))

                            // SIMD-accelerated bias correction
                            // Create mask for negative A values (only first 32 bytes matter, upper 32 already zero)
                            __mmask64 neg_mask = _mm512_cmplt_epi8_mask(a_vec, zero);

                            // Create vector of 1s where A < 0, 0 elsewhere
                            __m512i ones = _mm512_set1_epi8(1);
                            __m512i a_mask_ones = _mm512_maskz_mov_epi8(neg_mask, ones);

                            // Use VNNI to compute Σ(B[i] where A[i] < 0)
                            // dpbusd: 1 × B[i] (unsigned 1 × signed B) = B[i] where mask is true, 0 elsewhere
                            __m512i bias_vec = _mm512_dpbusd_epi32(zero, a_mask_ones, b_vec);

                            // Horizontal sum of first 8 INT32 lanes using AVX512 reduce (SIMD-optimized!)
                            int32_t sum_b_where_a_negative = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec);

                            int32_t corrected_dot = unsigned_sum - 256 * sum_b_where_a_negative;

                            // Apply scales immediately and accumulate to FP32 (matches llama.cpp)
                            fp_acc_[i * TILE_N + j] += static_cast<float>(corrected_dot) * a_scale * b_scale;
                        }
                    }
                }

                /**
                 * @brief AVX512-VNNI optimized accumulation for k_panel=64 (2 Q8_0 blocks)
                 *
                 * Processes 2 blocks (64 bytes) at once, using:
                 * - Full 64-byte SIMD register loads (no masking!)
                 * - All 16 INT32 lanes from DPBUSD (lanes 0-7 for block 0, lanes 8-15 for block 1)
                 * - 2× throughput compared to 32-byte version
                 *
                 * Uses _mm512_dpbusd_epi32(acc, a, b) which computes: acc += UNSIGNED(a) × SIGNED(b)
                 * Bias correction applied separately for each block.
                 */
                inline void accumulate_vnni_64_with_scales(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    const float *a_scales,
                    const float *b_scales)
                {
                    const __mmask16 lane_mask_8 = 0xFF;     // Mask for first 8 INT32 lanes (0-7)
                    const __mmask16 lane_mask_hi8 = 0xFF00; // Mask for upper 8 INT32 lanes (8-15)
                    __m512i zero = _mm512_setzero_si512();
                    __m512i ones = _mm512_set1_epi8(1);

                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * 64;                // 2 blocks = 64 bytes
                        float a_scale_0 = a_scales[i * K_BLOCKS_PER_ITER];     // Block 0 scale
                        float a_scale_1 = a_scales[i * K_BLOCKS_PER_ITER + 1]; // Block 1 scale

                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * 64;                // 2 blocks = 64 bytes
                            float b_scale_0 = b_scales[j * K_BLOCKS_PER_ITER];     // Block 0 scale
                            float b_scale_1 = b_scales[j * K_BLOCKS_PER_ITER + 1]; // Block 1 scale

                            // Load 64 INT8 elements (full register, no masking needed!)
                            __m512i a_vec = _mm512_loadu_si512(a_row);
                            __m512i b_vec = _mm512_loadu_si512(b_col);

                            // VNNI: UNSIGNED(a) × SIGNED(b)
                            // Result has 16 INT32 lanes:
                            //   - Lanes 0-7:  Dot product of bytes 0-31 (block 0)
                            //   - Lanes 8-15: Dot product of bytes 32-63 (block 1)
                            __m512i result = _mm512_dpbusd_epi32(zero, a_vec, b_vec);

                            // Horizontal reduction: sum lanes 0-7 for block 0, lanes 8-15 for block 1
                            int32_t unsigned_sum_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, result);
                            int32_t unsigned_sum_1 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result);

                            // Bias correction for BLOCK 0 (bytes 0-31)
                            // Extract lower 32 bytes for block 0
                            __m256i a_lo = _mm512_castsi512_si256(a_vec);
                            __m256i b_lo = _mm512_castsi512_si256(b_vec);
                            __m512i a_lo_512 = _mm512_castsi256_si512(a_lo);
                            __m512i b_lo_512 = _mm512_castsi256_si512(b_lo);

                            __mmask64 neg_mask_0 = _mm512_cmplt_epi8_mask(a_lo_512, zero);
                            __m512i a_mask_ones_0 = _mm512_maskz_mov_epi8(neg_mask_0, ones);
                            __m512i bias_vec_0 = _mm512_dpbusd_epi32(zero, a_mask_ones_0, b_lo_512);
                            int32_t sum_b_neg_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_0);

                            int32_t corrected_dot_0 = unsigned_sum_0 - 256 * sum_b_neg_0;

                            // Bias correction for BLOCK 1 (bytes 32-63)
                            // Extract upper 32 bytes for block 1
                            __m256i a_hi = _mm512_extracti64x4_epi64(a_vec, 1);
                            __m256i b_hi = _mm512_extracti64x4_epi64(b_vec, 1);
                            __m512i a_hi_512 = _mm512_castsi256_si512(a_hi);
                            __m512i b_hi_512 = _mm512_castsi256_si512(b_hi);

                            __mmask64 neg_mask_1 = _mm512_cmplt_epi8_mask(a_hi_512, zero);
                            __m512i a_mask_ones_1 = _mm512_maskz_mov_epi8(neg_mask_1, ones);
                            __m512i bias_vec_1 = _mm512_dpbusd_epi32(zero, a_mask_ones_1, b_hi_512);
                            int32_t sum_b_neg_1 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_1);

                            int32_t corrected_dot_1 = unsigned_sum_1 - 256 * sum_b_neg_1;

                            // Apply scales and accumulate both blocks to FP32 (matches llama.cpp)
                            // Split into two separate += for consistent rounding with 128-byte mode
                            fp_acc_[i * TILE_N + j] += static_cast<float>(corrected_dot_0) * a_scale_0 * b_scale_0;
                            fp_acc_[i * TILE_N + j] += static_cast<float>(corrected_dot_1) * a_scale_1 * b_scale_1;
                        }
                    }
                }
#endif /**                                                                                  \
        * @brief VNNI accumulation for 128 bytes (4 Q8_0 blocks)                            \
        *                                                                                   \
        * Processes 4 consecutive Q8_0 blocks (128 bytes total) using 2× AVX512 registers. \
        * Maximum throughput by fully utilizing instruction-level parallelism.              \
        *                                                                                   \
        * Layout:                                                                           \
        * - Bytes 0-31:   Block 0 (lanes 0-7 in first DPBUSD)                               \
        * - Bytes 32-63:  Block 1 (lanes 8-15 in first DPBUSD)                              \
        * - Bytes 64-95:  Block 2 (lanes 0-7 in second DPBUSD)                              \
        * - Bytes 96-127: Block 3 (lanes 8-15 in second DPBUSD)                             \
        */
                inline void accumulate_vnni_128_with_scales(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    const float *a_scales,
                    const float *b_scales)
                {
#if defined(__AVX512VNNI__)
                    const __mmask16 lane_mask_8 = 0xFF;     // Mask for first 8 INT32 lanes (0-7)
                    const __mmask16 lane_mask_hi8 = 0xFF00; // Mask for upper 8 INT32 lanes (8-15)
                    __m512i zero = _mm512_setzero_si512();
                    __m512i ones = _mm512_set1_epi8(1);

                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * 128;               // 4 blocks = 128 bytes
                        float a_scale_0 = a_scales[i * K_BLOCKS_PER_ITER];     // Block 0 scale
                        float a_scale_1 = a_scales[i * K_BLOCKS_PER_ITER + 1]; // Block 1 scale
                        float a_scale_2 = a_scales[i * K_BLOCKS_PER_ITER + 2]; // Block 2 scale
                        float a_scale_3 = a_scales[i * K_BLOCKS_PER_ITER + 3]; // Block 3 scale

                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * 128;               // 4 blocks = 128 bytes
                            float b_scale_0 = b_scales[j * K_BLOCKS_PER_ITER];     // Block 0 scale
                            float b_scale_1 = b_scales[j * K_BLOCKS_PER_ITER + 1]; // Block 1 scale
                            float b_scale_2 = b_scales[j * K_BLOCKS_PER_ITER + 2]; // Block 2 scale
                            float b_scale_3 = b_scales[j * K_BLOCKS_PER_ITER + 3]; // Block 3 scale

                            // ==========================================
                            // FIRST 64 BYTES (Blocks 0 and 1)
                            // ==========================================

                            __m512i a_vec_lo = _mm512_loadu_si512(a_row);
                            __m512i b_vec_lo = _mm512_loadu_si512(b_col);

                            // DPBUSD: 64 bytes → 16 INT32 lanes
                            __m512i result_lo = _mm512_dpbusd_epi32(zero, a_vec_lo, b_vec_lo);

                            // Extract block 0 (bytes 0-31, lanes 0-7)
                            int dot_unsigned_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, result_lo);

                            // Extract block 1 (bytes 32-63, lanes 8-15)
                            int dot_unsigned_1 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result_lo);

                            // Bias correction for block 0 (bytes 0-31)
                            __m256i a_lo = _mm512_castsi512_si256(a_vec_lo);
                            __m256i b_lo = _mm512_castsi512_si256(b_vec_lo);
                            __m512i a_lo_512 = _mm512_castsi256_si512(a_lo);
                            __m512i b_lo_512 = _mm512_castsi256_si512(b_lo);
                            __mmask64 neg_mask_0 = _mm512_cmplt_epi8_mask(a_lo_512, zero);
                            __m512i a_mask_ones_0 = _mm512_maskz_mov_epi8(neg_mask_0, ones);
                            __m512i bias_vec_0 = _mm512_dpbusd_epi32(zero, a_mask_ones_0, b_lo_512);
                            int sum_b_neg_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_0);
                            int dot_signed_0 = dot_unsigned_0 - 256 * sum_b_neg_0;

                            // Bias correction for block 1 (bytes 32-63)
                            __m256i a_hi = _mm512_extracti64x4_epi64(a_vec_lo, 1);
                            __m256i b_hi = _mm512_extracti64x4_epi64(b_vec_lo, 1);
                            __m512i a_hi_512 = _mm512_castsi256_si512(a_hi);
                            __m512i b_hi_512 = _mm512_castsi256_si512(b_hi);
                            __mmask64 neg_mask_1 = _mm512_cmplt_epi8_mask(a_hi_512, zero);
                            __m512i a_mask_ones_1 = _mm512_maskz_mov_epi8(neg_mask_1, ones);
                            __m512i bias_vec_1 = _mm512_dpbusd_epi32(zero, a_mask_ones_1, b_hi_512);
                            int sum_b_neg_1 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_1);
                            int dot_signed_1 = dot_unsigned_1 - 256 * sum_b_neg_1;

                            // ==========================================
                            // SECOND 64 BYTES (Blocks 2 and 3)
                            // ==========================================

                            __m512i a_vec_hi = _mm512_loadu_si512(a_row + 64);
                            __m512i b_vec_hi = _mm512_loadu_si512(b_col + 64);

                            // DPBUSD: 64 bytes → 16 INT32 lanes
                            __m512i result_hi = _mm512_dpbusd_epi32(zero, a_vec_hi, b_vec_hi);

                            // Extract block 2 (bytes 64-95, lanes 0-7)
                            int dot_unsigned_2 = _mm512_mask_reduce_add_epi32(lane_mask_8, result_hi);

                            // Extract block 3 (bytes 96-127, lanes 8-15)
                            int dot_unsigned_3 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result_hi);

                            // Bias correction for block 2 (bytes 64-95)
                            __m256i a_lo2 = _mm512_castsi512_si256(a_vec_hi);
                            __m256i b_lo2 = _mm512_castsi512_si256(b_vec_hi);
                            __m512i a_lo2_512 = _mm512_castsi256_si512(a_lo2);
                            __m512i b_lo2_512 = _mm512_castsi256_si512(b_lo2);
                            __mmask64 neg_mask_2 = _mm512_cmplt_epi8_mask(a_lo2_512, zero);
                            __m512i a_mask_ones_2 = _mm512_maskz_mov_epi8(neg_mask_2, ones);
                            __m512i bias_vec_2 = _mm512_dpbusd_epi32(zero, a_mask_ones_2, b_lo2_512);
                            int sum_b_neg_2 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_2);
                            int dot_signed_2 = dot_unsigned_2 - 256 * sum_b_neg_2;

                            // Bias correction for block 3 (bytes 96-127)
                            __m256i a_hi2 = _mm512_extracti64x4_epi64(a_vec_hi, 1);
                            __m256i b_hi2 = _mm512_extracti64x4_epi64(b_vec_hi, 1);
                            __m512i a_hi2_512 = _mm512_castsi256_si512(a_hi2);
                            __m512i b_hi2_512 = _mm512_castsi256_si512(b_hi2);
                            __mmask64 neg_mask_3 = _mm512_cmplt_epi8_mask(a_hi2_512, zero);
                            __m512i a_mask_ones_3 = _mm512_maskz_mov_epi8(neg_mask_3, ones);
                            __m512i bias_vec_3 = _mm512_dpbusd_epi32(zero, a_mask_ones_3, b_hi2_512);
                            int sum_b_neg_3 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_3);
                            int dot_signed_3 = dot_unsigned_3 - 256 * sum_b_neg_3;

                            // ==========================================
                            // ACCUMULATE WITH SCALES
                            // ==========================================

                            // Accumulate all 4 blocks with scales to FP32 (matches llama.cpp)
                            fp_acc_[i * TILE_N + j] += dot_signed_0 * a_scale_0 * b_scale_0;
                            fp_acc_[i * TILE_N + j] += dot_signed_1 * a_scale_1 * b_scale_1;
                            fp_acc_[i * TILE_N + j] += dot_signed_2 * a_scale_2 * b_scale_2;
                            fp_acc_[i * TILE_N + j] += dot_signed_3 * a_scale_3 * b_scale_3;
                        }
                    }
#endif
                }

                /**
                 * @brief VNNI accumulation for 256 bytes (8 Q8_0 blocks)
                 *
                 * Processes 8 consecutive Q8_0 blocks (256 bytes total) using 4× AVX512 registers.
                 * Maximum memory bandwidth utilization (full cache line).
                 *
                 * Layout:
                 * - Bytes 0-63:    Blocks 0-1 (first register pair)
                 * - Bytes 64-127:  Blocks 2-3 (second register pair)
                 * - Bytes 128-191: Blocks 4-5 (third register pair)
                 * - Bytes 192-255: Blocks 6-7 (fourth register pair)
                 */
                inline void accumulate_vnni_256_with_scales(
                    const int8_t *A_panel,
                    const int8_t *B_panel,
                    const float *a_scales,
                    const float *b_scales)
                {
#if defined(__AVX512VNNI__)
                    const __mmask16 lane_mask_8 = 0xFF;     // Mask for first 8 INT32 lanes (0-7)
                    const __mmask16 lane_mask_hi8 = 0xFF00; // Mask for upper 8 INT32 lanes (8-15)
                    __m512i zero = _mm512_setzero_si512();
                    __m512i ones = _mm512_set1_epi8(1);

                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * 256; // 8 blocks = 256 bytes

                        // Load all 8 scales for this row
                        float a_scale_0 = a_scales[i * K_BLOCKS_PER_ITER];
                        float a_scale_1 = a_scales[i * K_BLOCKS_PER_ITER + 1];
                        float a_scale_2 = a_scales[i * K_BLOCKS_PER_ITER + 2];
                        float a_scale_3 = a_scales[i * K_BLOCKS_PER_ITER + 3];
                        float a_scale_4 = a_scales[i * K_BLOCKS_PER_ITER + 4];
                        float a_scale_5 = a_scales[i * K_BLOCKS_PER_ITER + 5];
                        float a_scale_6 = a_scales[i * K_BLOCKS_PER_ITER + 6];
                        float a_scale_7 = a_scales[i * K_BLOCKS_PER_ITER + 7];

                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * 256; // 8 blocks = 256 bytes

                            // Load all 8 scales for this column
                            float b_scale_0 = b_scales[j * K_BLOCKS_PER_ITER];
                            float b_scale_1 = b_scales[j * K_BLOCKS_PER_ITER + 1];
                            float b_scale_2 = b_scales[j * K_BLOCKS_PER_ITER + 2];
                            float b_scale_3 = b_scales[j * K_BLOCKS_PER_ITER + 3];
                            float b_scale_4 = b_scales[j * K_BLOCKS_PER_ITER + 4];
                            float b_scale_5 = b_scales[j * K_BLOCKS_PER_ITER + 5];
                            float b_scale_6 = b_scales[j * K_BLOCKS_PER_ITER + 6];
                            float b_scale_7 = b_scales[j * K_BLOCKS_PER_ITER + 7];

                            // ==========================================
                            // FIRST 64 BYTES (Blocks 0 and 1)
                            // ==========================================

                            __m512i a_vec_0 = _mm512_loadu_si512(a_row);
                            __m512i b_vec_0 = _mm512_loadu_si512(b_col);
                            __m512i result_0 = _mm512_dpbusd_epi32(zero, a_vec_0, b_vec_0);

                            int dot_unsigned_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, result_0);
                            int dot_unsigned_1 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result_0);

                            // Bias correction block 0
                            __m256i a_lo_0 = _mm512_castsi512_si256(a_vec_0);
                            __m256i b_lo_0 = _mm512_castsi512_si256(b_vec_0);
                            __m512i a_lo_0_512 = _mm512_castsi256_si512(a_lo_0);
                            __m512i b_lo_0_512 = _mm512_castsi256_si512(b_lo_0);
                            __mmask64 neg_mask_0 = _mm512_cmplt_epi8_mask(a_lo_0_512, zero);
                            __m512i a_mask_ones_0 = _mm512_maskz_mov_epi8(neg_mask_0, ones);
                            __m512i bias_vec_0 = _mm512_dpbusd_epi32(zero, a_mask_ones_0, b_lo_0_512);
                            int sum_b_neg_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_0);
                            int dot_signed_0 = dot_unsigned_0 - 256 * sum_b_neg_0;

                            // Bias correction block 1
                            __m256i a_hi_0 = _mm512_extracti64x4_epi64(a_vec_0, 1);
                            __m256i b_hi_0 = _mm512_extracti64x4_epi64(b_vec_0, 1);
                            __m512i a_hi_0_512 = _mm512_castsi256_si512(a_hi_0);
                            __m512i b_hi_0_512 = _mm512_castsi256_si512(b_hi_0);
                            __mmask64 neg_mask_1 = _mm512_cmplt_epi8_mask(a_hi_0_512, zero);
                            __m512i a_mask_ones_1 = _mm512_maskz_mov_epi8(neg_mask_1, ones);
                            __m512i bias_vec_1 = _mm512_dpbusd_epi32(zero, a_mask_ones_1, b_hi_0_512);
                            int sum_b_neg_1 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_1);
                            int dot_signed_1 = dot_unsigned_1 - 256 * sum_b_neg_1;

                            // ==========================================
                            // SECOND 64 BYTES (Blocks 2 and 3)
                            // ==========================================

                            __m512i a_vec_1 = _mm512_loadu_si512(a_row + 64);
                            __m512i b_vec_1 = _mm512_loadu_si512(b_col + 64);
                            __m512i result_1 = _mm512_dpbusd_epi32(zero, a_vec_1, b_vec_1);

                            int dot_unsigned_2 = _mm512_mask_reduce_add_epi32(lane_mask_8, result_1);
                            int dot_unsigned_3 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result_1);

                            // Bias correction block 2
                            __m256i a_lo_1 = _mm512_castsi512_si256(a_vec_1);
                            __m256i b_lo_1 = _mm512_castsi512_si256(b_vec_1);
                            __m512i a_lo_1_512 = _mm512_castsi256_si512(a_lo_1);
                            __m512i b_lo_1_512 = _mm512_castsi256_si512(b_lo_1);
                            __mmask64 neg_mask_2 = _mm512_cmplt_epi8_mask(a_lo_1_512, zero);
                            __m512i a_mask_ones_2 = _mm512_maskz_mov_epi8(neg_mask_2, ones);
                            __m512i bias_vec_2 = _mm512_dpbusd_epi32(zero, a_mask_ones_2, b_lo_1_512);
                            int sum_b_neg_2 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_2);
                            int dot_signed_2 = dot_unsigned_2 - 256 * sum_b_neg_2;

                            // Bias correction block 3
                            __m256i a_hi_1 = _mm512_extracti64x4_epi64(a_vec_1, 1);
                            __m256i b_hi_1 = _mm512_extracti64x4_epi64(b_vec_1, 1);
                            __m512i a_hi_1_512 = _mm512_castsi256_si512(a_hi_1);
                            __m512i b_hi_1_512 = _mm512_castsi256_si512(b_hi_1);
                            __mmask64 neg_mask_3 = _mm512_cmplt_epi8_mask(a_hi_1_512, zero);
                            __m512i a_mask_ones_3 = _mm512_maskz_mov_epi8(neg_mask_3, ones);
                            __m512i bias_vec_3 = _mm512_dpbusd_epi32(zero, a_mask_ones_3, b_hi_1_512);
                            int sum_b_neg_3 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_3);
                            int dot_signed_3 = dot_unsigned_3 - 256 * sum_b_neg_3;

                            // ==========================================
                            // THIRD 64 BYTES (Blocks 4 and 5)
                            // ==========================================

                            __m512i a_vec_2 = _mm512_loadu_si512(a_row + 128);
                            __m512i b_vec_2 = _mm512_loadu_si512(b_col + 128);
                            __m512i result_2 = _mm512_dpbusd_epi32(zero, a_vec_2, b_vec_2);

                            int dot_unsigned_4 = _mm512_mask_reduce_add_epi32(lane_mask_8, result_2);
                            int dot_unsigned_5 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result_2);

                            // Bias correction block 4
                            __m256i a_lo_2 = _mm512_castsi512_si256(a_vec_2);
                            __m256i b_lo_2 = _mm512_castsi512_si256(b_vec_2);
                            __m512i a_lo_2_512 = _mm512_castsi256_si512(a_lo_2);
                            __m512i b_lo_2_512 = _mm512_castsi256_si512(b_lo_2);
                            __mmask64 neg_mask_4 = _mm512_cmplt_epi8_mask(a_lo_2_512, zero);
                            __m512i a_mask_ones_4 = _mm512_maskz_mov_epi8(neg_mask_4, ones);
                            __m512i bias_vec_4 = _mm512_dpbusd_epi32(zero, a_mask_ones_4, b_lo_2_512);
                            int sum_b_neg_4 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_4);
                            int dot_signed_4 = dot_unsigned_4 - 256 * sum_b_neg_4;

                            // Bias correction block 5
                            __m256i a_hi_2 = _mm512_extracti64x4_epi64(a_vec_2, 1);
                            __m256i b_hi_2 = _mm512_extracti64x4_epi64(b_vec_2, 1);
                            __m512i a_hi_2_512 = _mm512_castsi256_si512(a_hi_2);
                            __m512i b_hi_2_512 = _mm512_castsi256_si512(b_hi_2);
                            __mmask64 neg_mask_5 = _mm512_cmplt_epi8_mask(a_hi_2_512, zero);
                            __m512i a_mask_ones_5 = _mm512_maskz_mov_epi8(neg_mask_5, ones);
                            __m512i bias_vec_5 = _mm512_dpbusd_epi32(zero, a_mask_ones_5, b_hi_2_512);
                            int sum_b_neg_5 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_5);
                            int dot_signed_5 = dot_unsigned_5 - 256 * sum_b_neg_5;

                            // ==========================================
                            // FOURTH 64 BYTES (Blocks 6 and 7)
                            // ==========================================

                            __m512i a_vec_3 = _mm512_loadu_si512(a_row + 192);
                            __m512i b_vec_3 = _mm512_loadu_si512(b_col + 192);
                            __m512i result_3 = _mm512_dpbusd_epi32(zero, a_vec_3, b_vec_3);

                            int dot_unsigned_6 = _mm512_mask_reduce_add_epi32(lane_mask_8, result_3);
                            int dot_unsigned_7 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result_3);

                            // Bias correction block 6
                            __m256i a_lo_3 = _mm512_castsi512_si256(a_vec_3);
                            __m256i b_lo_3 = _mm512_castsi512_si256(b_vec_3);
                            __m512i a_lo_3_512 = _mm512_castsi256_si512(a_lo_3);
                            __m512i b_lo_3_512 = _mm512_castsi256_si512(b_lo_3);
                            __mmask64 neg_mask_6 = _mm512_cmplt_epi8_mask(a_lo_3_512, zero);
                            __m512i a_mask_ones_6 = _mm512_maskz_mov_epi8(neg_mask_6, ones);
                            __m512i bias_vec_6 = _mm512_dpbusd_epi32(zero, a_mask_ones_6, b_lo_3_512);
                            int sum_b_neg_6 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_6);
                            int dot_signed_6 = dot_unsigned_6 - 256 * sum_b_neg_6;

                            // Bias correction block 7
                            __m256i a_hi_3 = _mm512_extracti64x4_epi64(a_vec_3, 1);
                            __m256i b_hi_3 = _mm512_extracti64x4_epi64(b_vec_3, 1);
                            __m512i a_hi_3_512 = _mm512_castsi256_si512(a_hi_3);
                            __m512i b_hi_3_512 = _mm512_castsi256_si512(b_hi_3);
                            __mmask64 neg_mask_7 = _mm512_cmplt_epi8_mask(a_hi_3_512, zero);
                            __m512i a_mask_ones_7 = _mm512_maskz_mov_epi8(neg_mask_7, ones);
                            __m512i bias_vec_7 = _mm512_dpbusd_epi32(zero, a_mask_ones_7, b_hi_3_512);
                            int sum_b_neg_7 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec_7);
                            int dot_signed_7 = dot_unsigned_7 - 256 * sum_b_neg_7;

                            // ==========================================
                            // ACCUMULATE WITH SCALES
                            // ==========================================

                            // Accumulate all 8 blocks with scales to FP32 (matches llama.cpp)
                            fp_acc_[i * TILE_N + j] += dot_signed_0 * a_scale_0 * b_scale_0;
                            fp_acc_[i * TILE_N + j] += dot_signed_1 * a_scale_1 * b_scale_1;
                            fp_acc_[i * TILE_N + j] += dot_signed_2 * a_scale_2 * b_scale_2;
                            fp_acc_[i * TILE_N + j] += dot_signed_3 * a_scale_3 * b_scale_3;
                            fp_acc_[i * TILE_N + j] += dot_signed_4 * a_scale_4 * b_scale_4;
                            fp_acc_[i * TILE_N + j] += dot_signed_5 * a_scale_5 * b_scale_5;
                            fp_acc_[i * TILE_N + j] += dot_signed_6 * a_scale_6 * b_scale_6;
                            fp_acc_[i * TILE_N + j] += dot_signed_7 * a_scale_7 * b_scale_7;
                        }
                    }
#endif
                }

                // ============================================================
                // ZERO-COPY VNNI IMPLEMENTATIONS (NO memcpy!)
                // ============================================================

#if defined(__AVX512VNNI__)
                /**
                 * @brief Zero-copy VNNI 32-byte (single block) - works directly from Q8_0 blocks
                 */
                template <typename BlockProvider>
                inline void accumulate_vnni_32_zerocopy(
                    const Q8_0Block *A_blocks,
                    BlockProvider &B_provider,
                    int tile_m,
                    int tile_n,
                    int ii,
                    int jj,
                    size_t kb,
                    size_t k_blocks)
                {
                    const __mmask64 mask32 = 0x00000000FFFFFFFFULL;
                    const __mmask16 lane_mask_8 = 0xFF;
                    __m512i zero = _mm512_setzero_si512();
                    __m512i ones = _mm512_set1_epi8(1);

                    for (int i = 0; i < tile_m; ++i)
                    {
                        const Q8_0Block *a_block = &A_blocks[(ii + i) * k_blocks + kb];
                        float a_scale = fp16_to_fp32(a_block->d);

                        for (int j = 0; j < tile_n; ++j)
                        {
                            const Q8_0Block *b_block = B_provider.get_q8_block(jj + j, kb);
                            float b_scale = fp16_to_fp32(b_block->d);

                            // Load directly from source blocks (NO memcpy!)
                            __m512i a_vec = _mm512_maskz_loadu_epi8(mask32, a_block->qs);
                            __m512i b_vec = _mm512_maskz_loadu_epi8(mask32, b_block->qs);

                            __m512i result = _mm512_dpbusd_epi32(zero, a_vec, b_vec);
                            int32_t unsigned_sum = _mm512_mask_reduce_add_epi32(lane_mask_8, result);

                            // Bias correction
                            __mmask64 neg_mask = _mm512_cmplt_epi8_mask(a_vec, zero);
                            __m512i a_mask_ones = _mm512_maskz_mov_epi8(neg_mask, ones);
                            __m512i bias_vec = _mm512_dpbusd_epi32(zero, a_mask_ones, b_vec);
                            int32_t sum_b_neg = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_vec);

                            int32_t corrected_dot = unsigned_sum - 256 * sum_b_neg;
                            fp_acc_[i * TILE_N + j] += static_cast<float>(corrected_dot) * a_scale * b_scale;
                        }
                    }
                }

                /**
                 * @brief Zero-copy VNNI 64-byte (two blocks) - works directly from Q8_0 blocks
                 */
                template <typename BlockProvider>
                inline void accumulate_vnni_64_zerocopy(
                    const Q8_0Block *A_blocks,
                    BlockProvider &B_provider,
                    int tile_m,
                    int tile_n,
                    int ii,
                    int jj,
                    size_t kb,
                    size_t k_blocks)
                {
                    const __mmask16 lane_mask_8 = 0xFF;
                    const __mmask16 lane_mask_hi8 = 0xFF00;
                    __m512i zero = _mm512_setzero_si512();
                    __m512i ones = _mm512_set1_epi8(1);

                    for (int i = 0; i < tile_m; ++i)
                    {
                        const Q8_0Block *a_block_0 = &A_blocks[(ii + i) * k_blocks + kb];
                        const Q8_0Block *a_block_1 = &A_blocks[(ii + i) * k_blocks + kb + 1];
                        float a_scale_0 = fp16_to_fp32(a_block_0->d);
                        float a_scale_1 = fp16_to_fp32(a_block_1->d);

                        for (int j = 0; j < tile_n; ++j)
                        {
                            const Q8_0Block *b_block_0 = B_provider.get_q8_block(jj + j, kb);
                            const Q8_0Block *b_block_1 = B_provider.get_q8_block(jj + j, kb + 1);
                            float b_scale_0 = fp16_to_fp32(b_block_0->d);
                            float b_scale_1 = fp16_to_fp32(b_block_1->d);

                            // Load both blocks into 64-byte register (using unaligned loads directly from Q8_0 blocks)
                            __m256i a_lo_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block_0->qs));
                            __m256i a_hi_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block_1->qs));
                            __m512i a_vec = _mm512_inserti64x4(_mm512_castsi256_si512(a_lo_256), a_hi_256, 1);

                            __m256i b_lo_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_block_0->qs));
                            __m256i b_hi_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b_block_1->qs));
                            __m512i b_vec = _mm512_inserti64x4(_mm512_castsi256_si512(b_lo_256), b_hi_256, 1);

                            __m512i result = _mm512_dpbusd_epi32(zero, a_vec, b_vec);

                            // Extract dot products for both blocks
                            int dot_unsigned_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, result);
                            int dot_unsigned_1 = _mm512_mask_reduce_add_epi32(lane_mask_hi8, result);

                            // Bias correction for block 0
                            __m512i a_lo_512 = _mm512_castsi256_si512(a_lo_256);
                            __m512i b_lo_512 = _mm512_castsi256_si512(b_lo_256);
                            __mmask64 neg_mask_0 = _mm512_cmplt_epi8_mask(a_lo_512, zero);
                            __m512i a_mask_0 = _mm512_maskz_mov_epi8(neg_mask_0, ones);
                            __m512i bias_0 = _mm512_dpbusd_epi32(zero, a_mask_0, b_lo_512);
                            int sum_b_neg_0 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_0);
                            int dot_signed_0 = dot_unsigned_0 - 256 * sum_b_neg_0;

                            // Bias correction for block 1
                            __m512i a_hi_512 = _mm512_castsi256_si512(a_hi_256);
                            __m512i b_hi_512 = _mm512_castsi256_si512(b_hi_256);
                            __mmask64 neg_mask_1 = _mm512_cmplt_epi8_mask(a_hi_512, zero);
                            __m512i a_mask_1 = _mm512_maskz_mov_epi8(neg_mask_1, ones);
                            __m512i bias_1 = _mm512_dpbusd_epi32(zero, a_mask_1, b_hi_512);
                            int sum_b_neg_1 = _mm512_mask_reduce_add_epi32(lane_mask_8, bias_1);
                            int dot_signed_1 = dot_unsigned_1 - 256 * sum_b_neg_1;

                            // Accumulate with per-block scales
                            fp_acc_[i * TILE_N + j] +=
                                static_cast<float>(dot_signed_0) * a_scale_0 * b_scale_0 +
                                static_cast<float>(dot_signed_1) * a_scale_1 * b_scale_1;
                        }
                    }
                }

                /**
                 * @brief Zero-copy VNNI 128-byte (four blocks) - works directly from Q8_0 blocks
                 */
                template <typename BlockProvider>
                inline void accumulate_vnni_128_zerocopy(
                    const Q8_0Block *A_blocks,
                    BlockProvider &B_provider,
                    int tile_m,
                    int tile_n,
                    int ii,
                    int jj,
                    size_t kb,
                    size_t k_blocks)
                {
                    // For simplicity, call 64-byte version twice
                    accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks);
                    accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb + 2, k_blocks);
                }

                /**
                 * @brief Zero-copy VNNI 256-byte (eight blocks) - works directly from Q8_0 blocks
                 */
                template <typename BlockProvider>
                inline void accumulate_vnni_256_zerocopy(
                    const Q8_0Block *A_blocks,
                    BlockProvider &B_provider,
                    int tile_m,
                    int tile_n,
                    int ii,
                    int jj,
                    size_t kb,
                    size_t k_blocks)
                {
                    // For simplicity, call 64-byte version four times
                    accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb, k_blocks);
                    accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb + 2, k_blocks);
                    accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb + 4, k_blocks);
                    accumulate_vnni_64_zerocopy(A_blocks, B_provider, tile_m, tile_n, ii, jj, kb + 6, k_blocks);
                }
#endif // __AVX512VNNI__

                /**
                 * @brief Quantize FP32 array to Q8_0 block
                 *
                 * @param fp_values Input FP32 values (32 elements)
                 * @param block Output Q8_0 block
                 */
                static void quantize_fp32_to_q8_0(const float *fp_values, Q8_0Block *block)
                {
                    // Find max absolute value
                    float amax = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        float abs_val = std::fabs(fp_values[i]);
                        if (abs_val > amax)
                            amax = abs_val;
                    }

                    // Compute scale
                    float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                    float inv_scale = 1.0f / scale;

                    // Store scale as FP16
                    block->d = fp32_to_fp16(scale);

                    // Quantize to INT8 [-127, 127]
                    for (int i = 0; i < 32; ++i)
                    {
                        float scaled = fp_values[i] * inv_scale;
                        int32_t q = static_cast<int32_t>(std::lroundf(scaled));
                        q = std::max(-127, std::min(127, q));
                        block->qs[i] = static_cast<int8_t>(q);
                    }
                }

                // INT32 accumulator (for future VNNI optimizations)
                alignas(64) int32_t acc_[TILE_M * TILE_N];

                // FP32 accumulator (scales applied per K-block during accumulation)
                alignas(64) float fp_acc_[TILE_M * TILE_N];
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
