/**
 * @file Q8_1GemmKernel.h
 * @brief Q8_1 × Q8_0 → FP32 GEMM kernel using IQ8_1Decodable interface
 * @author David Sanftenberg
 *
 * Computes C_fp32 = A × B where A is any IActivationTensor (FP32/FP16/BF16/Q8_1)
 * and B is Q8_0 quantized tensor (weights).
 *
 * Key Innovation: Uses IQ8_1Decodable interface to convert activations on-the-fly
 * - FP32/FP16/BF16: Quantize to Q8_1 during GEMM (thread-local blocks)
 * - Q8_1Tensor: Direct pointer access (zero-copy)
 *
 * Q8_1 Format (Activations):
 * - Blocks of 32 int8 values
 * - Per-block FP16 scale (uint16_t d)
 * - Per-block FP16 pre-computed sum (uint16_t s = d × Σ(qs[i]))
 * - Block size: 36 bytes (2 + 2 + 32)
 *
 * Q8_0 Format (Weights):
 * - Blocks of 32 int8 values
 * - Per-block FP16 scale (uint16_t d)
 * - Block size: 34 bytes (2 + 32)
 *
 * Formula: C = (A_scales ⊗ B_scales) ⊙ (A_quants × B_quants - A_sums × 128)
 * Where A_sums are PRE-COMPUTED (eliminates 448 horizontal reductions per layer!)
 *
 * Microkernel: MR×NR (manageable register pressure, 2048 FLOPs per call)
 */

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "utils/DebugEnv.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar2
{

    // Forward declarations
    class IActivationTensor;

    /**
     * @class Q8_1GemmKernel
     * @brief High-performance Q8_1 × Q8_0 GEMM using IQ8_1Decodable interface
     *
     * Key Design Decisions:
     * 1. Generic activation input - accepts any IActivationTensor via IQ8_1Decodable
     * 2. Pre-computed sums - Q8_1 blocks include s = d × Σ(qs[i]) (eliminates 448 reductions!)
     * 3. On-the-fly quantization - FP32/FP16/BF16 quantized to Q8_1 during GEMM
     * 4. Zero-copy for Q8_1 - Q8_1Tensor returns direct pointer (no allocation)
     * 5. Parameterized microkernel (MR×NR template) - tunable for different architectures
     * 6. Scale packing - B scales packed alongside quantized values
     * 7. No A-packing - Q8_1 blocks accessed via IQ8_1Decodable (zero overhead for Q8_1Tensor)
     * 8. B packing - transpose + s8→u8 for VNNI layout + scale extraction
     * 9. NC/KC blocking - cache-friendly blocking for large matrices
     *
     * Autotuner Configuration Space:
     * ==============================
     * The template parameters form a configuration space that can be explored by autotuners
     * to find optimal settings for different problem sizes (M, N, K) and hardware platforms.
     *
     * Key Insight: Different optimizations perform better at different problem sizes!
     * - JR_UNROLL=4: +5.5% at M=4096, but -20% at M=512 (register pressure hurts small M)
     * - JR_UNROLL=2: Balanced performance across all M sizes (current default)
     *
     * Autotuner should select configuration based on problem characteristics:
     * - Small M (512-1024): Prefer low register pressure → JR_UNROLL=2, smaller MR/NR
     * - Medium M (2048-4096): Higher ILP beneficial → JR_UNROLL=4 can win
     * - Large M (8192+): Memory bandwidth dominates → JR_UNROLL has minimal impact
     *
     * @tparam MR_PARAM M register blocking (rows per microkernel)
     *   Range: [8, 16, 32] - Sweep found 32 optimal for most cases
     *   Trade-off: Larger = better A reuse, but higher register pressure
     *
     * @tparam NR_PARAM N register blocking (columns per microkernel)
     *   Range: [32, 64, 128] - Sweep found 128 optimal (+5.3% over 64)
     *   Trade-off: Larger = better B panel reuse, but more B data per microkernel
     *
     * @tparam PREFETCH_A_PARAM A block prefetch distance (0-5, default 4)
     *   Range: [0, 2, 4] - Empirically found 4 optimal
     *   Trade-off: Too low = cache misses, too high = pollutes cache
     *
     * @tparam NC_PARAM N cache blocking (0=auto, default 896 fits L2)
     *   Range: [0, 448, 896, 1792] - 0 means auto-select based on N
     *   Trade-off: Fits working set in L2 cache (B panel + C accumulator)
     *
     * @tparam KC_PARAM K cache blocking in K-blocks (0=auto, default 128 max storage)
     *   Range: [0, 28, 56, 128] - 0 means use all K_blocks (no blocking)
     *   Trade-off: Streaming GEMM removed accum_vec, so KC can now be larger
     *
     * @tparam JR_UNROLL_PARAM Column unrolling factor for dpbusd ILP (1, 2, 4, 6, or 8, default 2)
     *   Range: [1, 2, 4, 6, 8] - Controls inner loop unrolling
     *   Trade-off: Higher = more ILP but higher register pressure (hurts small M)
     *   Performance impact measured (Nov 2025):
     *     M=512:  JR=2: 242 GFLOPS, JR=4: 207 GFLOPS (-14%)
     *     M=4096: JR=2: 424 GFLOPS, JR=4: 470 GFLOPS (+11%)
     *     M=8192: JR=2: 452 GFLOPS, JR=4: 491 GFLOPS (+9%)
     *   Implementation: Template helper function with compile-time unrolling
     *     (eliminates code duplication - all unroll factors use same logic)
     */
    template <int MR_PARAM = 32, int NR_PARAM = 128, int PREFETCH_A_PARAM = 4,
              int NC_PARAM = 0, int KC_PARAM = 0, int JR_UNROLL_PARAM = 2>
    class Q8_1GemmKernelTemplate
    {
    public:
        static constexpr int MR = MR_PARAM;                 // M register blocking
        static constexpr int NR = NR_PARAM;                 // N register blocking
        static constexpr int BLOCK_SIZE = 32;               // Q8_0Block::BLOCK_SIZE
        static constexpr int PREFETCH_A = PREFETCH_A_PARAM; // A prefetch distance
        static constexpr int VECTOR_WIDTH = 16;             // K-blocks per iteration (MUST be 16 for __m512)
        static constexpr int NC = NC_PARAM;                 // N cache blocking (0=auto)
        static constexpr int KC = KC_PARAM;                 // K cache blocking in blocks (0=auto)
        static constexpr int JR_UNROLL = JR_UNROLL_PARAM;   // Column unrolling factor (2 or 4)

        /**
         * @brief Packed B panel structure (optimized for 64-byte ZMM loads)
         *
         * Layout for one panel (NR columns):
         * - quants: [kb][jr_pair][64] where jr_pair = jr/2 (2 columns per ZMM)
         * - scales[K_blocks][NR]: FP16 scales for each column/block combination
         *
         * Key optimization: Pack 2 columns per ZMM (no padding waste!)
         * - Columns 0,1 → ZMM[0]: [32 bytes col0][32 bytes col1]
         * - Columns 2,3 → ZMM[1]: [32 bytes col2][32 bytes col3]
         * - Columns 4,5 → ZMM[2]: [32 bytes col4][32 bytes col5]
         * - Columns 6,7 → ZMM[3]: [32 bytes col6][32 bytes col7]
         */
        struct PackedBPanel
        {
            std::vector<uint8_t> quants;  // [K_blocks][NR/2][64] (2 cols per ZMM)
            std::vector<uint16_t> scales; // [K_blocks*NR]
            int K_blocks;

            PackedBPanel(int k_blocks) : K_blocks(k_blocks)
            {
                // Layout: [kb][jr_pair][64 bytes] where each 64 bytes = 2 columns
                // Total: K_blocks × (NR/2) × 64 bytes
                size_t bytes_per_zmm = 64; // 2 columns × 32 bytes (no padding!)
                size_t total_bytes = k_blocks * (NR / 2) * bytes_per_zmm;
                quants.resize(total_bytes, 128); // Fill with zero-value (128 in u8)
                scales.resize(k_blocks * NR);
            }
        };

        // ====================================================================
        // UNIT-TESTABLE HELPER FUNCTIONS
        // ====================================================================

        /**
         * @brief Compute Q8_1 pre-computed sum from block
         * @param a_sum FP16 pre-computed sum (s = d × Σ(qs[i]))
         * @param a_scale FP16 scale factor
         * @return Σ(qs[i]) = s / d
         */
        static inline float compute_sum_qs(uint16_t a_sum_fp16, uint16_t a_scale_fp16)
        {
            float sum_a = fp16_to_fp32(a_sum_fp16);
            float a_scale = fp16_to_fp32(a_scale_fp16);
            // Add epsilon to avoid division by zero for zero blocks
            return sum_a / std::max(a_scale, 1e-10f);
        }

        /**
         * @brief Apply Q8_0 compensation formula
         *
         * Q8_0 uses SYMMETRIC quantization (range [-127, 127]) with NO zero-point offset.
         * Unlike asymmetric quantization schemes, Q8_0 does NOT require 128 compensation.
         *
         * The original formula `accum - sum_qs*128` was incorrect and caused sign errors
         * (208% error on test data). Correct formula is simply: accum (no compensation).
         *
         * @param accum Raw dpbusd accumulator (Σ(a_qs[i] * b_qs[i]))
         * @param sum_qs Sum of A quantized values (Σ(a_qs[i])) - unused for Q8_0
         * @return Unmodified accumulator (Q8_0 symmetric quantization needs no compensation)
         */
        static inline int32_t apply_compensation(int32_t accum, float sum_qs)
        {
            // Q8_0 is symmetric quantization - no compensation needed!
            (void)sum_qs; // Unused for Q8_0, kept for API compatibility
            return accum;
        }

        /**
         * @brief Scale compensated value to FP32
         * @param compensated Compensated int32 value
         * @param a_scale A block scale (FP16 converted to FP32)
         * @param b_scale B block scale (FP16 converted to FP32)
         * @return FP32 result (compensated * a_scale * b_scale)
         */
        static inline float scale_to_fp32(int32_t compensated, float a_scale, float b_scale)
        {
            return static_cast<float>(compensated) * a_scale * b_scale;
        }

        /**
         * @brief Complete Q8_1×Q8_0 block computation (single block pair)
         * @param accum Raw dpbusd result
         * @param a_sum_fp16 Q8_1 pre-computed sum (FP16)
         * @param a_scale_fp16 A block scale (FP16)
         * @param b_scale_fp16 B block scale (FP16)
         * @return FP32 result for this block pair
         */
        static inline float compute_block_result(int32_t accum,
                                                 uint16_t a_sum_fp16,
                                                 uint16_t a_scale_fp16,
                                                 uint16_t b_scale_fp16)
        {
            float sum_qs = compute_sum_qs(a_sum_fp16, a_scale_fp16);
            int32_t compensated = apply_compensation(accum, sum_qs);
            float a_scale = fp16_to_fp32(a_scale_fp16);
            float b_scale = fp16_to_fp32(b_scale_fp16);
            return scale_to_fp32(compensated, a_scale, b_scale);
        }

        /**
         * @brief Compute block index in column-major layout
         * @param row Row index in B (column in original K×N matrix)
         * @param k_block K-block index
         * @param K_blocks Total number of K-blocks
         * @return Linear block index
         */
        static inline size_t block_index_col_major(size_t row, size_t k_block, size_t K_blocks)
        {
            return row * K_blocks + k_block;
        }

        /**
         * @brief Unpack single B column value from packed layout (SCALAR VERSION)
         *
         * Packed layout: [kb][jr_pair][64] where 64 bytes = [col0_bytes0..31][col1_bytes0..31]
         *
         * @param B_quants Packed quantized values
         * @param kb K-block index
         * @param jr Column index (0..NR-1)
         * @param k_in Element within block (0..31)
         * @return Unpacked int8 value (converted from uint8 storage)
         */
        static inline int8_t unpack_B_value(const uint8_t *B_quants, int kb, int jr, int k_in)
        {
            // Determine which jr_pair contains column jr
            int jr_pair = jr / 2;
            int col_offset = (jr % 2) * 32; // 0 for even jr, 32 for odd jr

            // Calculate base address for this jr_pair's 64-byte ZMM
            const uint8_t *zmm_base = B_quants + (kb * (NR / 2) + jr_pair) * 64;

            // Read value and convert from uint8 (0-255) to int8 (-128 to 127)
            uint8_t b_val_u8 = zmm_base[col_offset + k_in];
            return static_cast<int8_t>(b_val_u8 - 128);
        }

        /**
         * @brief Verify B packing/unpacking round-trip correctness
         *
         * Tests that pack_B_panel() followed by unpack_B_value() returns original data.
         *
         * @param original_blocks Original Q8_0 blocks [N][K_blocks] (column-major)
         * @param packed Packed B panel
         * @param N Number of columns
         * @param K_blocks Number of K-blocks
         * @return true if all values match, false otherwise
         */
        static inline bool verify_B_packing(const Q8_0Block *original_blocks,
                                            const PackedBPanel &packed,
                                            int N, int K_blocks)
        {
            for (int jr = 0; jr < N; ++jr)
            {
                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    const Q8_0Block &orig_block = original_blocks[jr * K_blocks + kb];

                    // Verify each quantized value
                    for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                    {
                        int8_t original = orig_block.qs[k_in];
                        int8_t unpacked = unpack_B_value(packed.quants.data(), kb, jr, k_in);

                        if (original != unpacked)
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        /**
         * @brief Compute per-block contribution (SCALAR VERSION for testing vectorized path)
         *
         * This is what the vectorized path should compute for each K-block:
         * contribution = accum[kb] * a_scale[kb] * b_scale[kb]
         *
         * @param accum_kb Accumulator for this K-block (dpbusd result)
         * @param a_scale_fp16 A scale for this K-block (FP16)
         * @param b_scale_fp16 B scale for this K-block (FP16)
         * @return FP32 contribution for this K-block
         */
        static inline float compute_per_block_contribution(int32_t accum_kb,
                                                           uint16_t a_scale_fp16,
                                                           uint16_t b_scale_fp16)
        {
            // Q8_0 symmetric quantization: NO compensation needed!
            int32_t compensated = accum_kb;

            // Scale to FP32
            float a_scale = fp16_to_fp32(a_scale_fp16);
            float b_scale = fp16_to_fp32(b_scale_fp16);

            return static_cast<float>(compensated) * a_scale * b_scale;
        }

        /**
         * @brief Compute GEMM result for single output element (sum across K-blocks)
         *
         * This is the reference implementation that the vectorized path should match.
         *
         * @param accum_values Array of accumulator values [K_blocks]
         * @param a_scales Array of A scales [K_blocks] (FP16)
         * @param b_scales Array of B scales [K_blocks] (FP16)
         * @param K_blocks Number of K-blocks
         * @return FP32 result (sum of per-block contributions)
         */
        static inline float compute_gemm_result_scalar(const int32_t *accum_values,
                                                       const uint16_t *a_scales,
                                                       const uint16_t *b_scales,
                                                       int K_blocks)
        {
            float result = 0.0f;
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                result += compute_per_block_contribution(accum_values[kb], a_scales[kb], b_scales[kb]);
            }
            return result;
        }

        /**
         * @brief Verify vectorized post-processing against scalar reference
         *
         * Tests that the vectorized reduction produces the same result as scalar accumulation.
         * This isolates potential bugs in the SIMD horizontal reduction logic.
         *
         * @param accum_values Accumulator values for one output element [K_blocks]
         * @param a_scales A scales for one row [K_blocks] (FP16)
         * @param b_scales B scales for one column [K_blocks] (FP16)
         * @param K_blocks Number of K-blocks
         * @return true if vectorized and scalar results match within tolerance
         */
        static inline bool verify_vectorized_reduction(const int32_t *accum_values,
                                                       const uint16_t *a_scales,
                                                       const uint16_t *b_scales,
                                                       int K_blocks)
        {
            // Compute scalar reference
            float scalar_result = compute_gemm_result_scalar(accum_values, a_scales, b_scales, K_blocks);

            // Compute vectorized result (simulate what the kernel does)
            float vectorized_result = 0.0f;

            // Process in chunks of 16 (VECTOR_WIDTH for AVX-512)
            constexpr int VECTOR_WIDTH = 16;
            int kb = 0;

            for (; kb + VECTOR_WIDTH <= K_blocks; kb += VECTOR_WIDTH)
            {
                // Load 16 accumulators and convert to FP32
                __m512i accum_i32[1];
                accum_i32[0] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&accum_values[kb]));
                __m512 accum_f32 = _mm512_cvtepi32_ps(accum_i32[0]);

                // Load 16 A scales (FP16) and convert to FP32
                __m256i a_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a_scales[kb]));
                __m512 a_scales_f32 = _mm512_cvtph_ps(a_scales_fp16);

                // Load 16 B scales (FP16) and convert to FP32
                __m256i b_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b_scales[kb]));
                __m512 b_scales_f32 = _mm512_cvtph_ps(b_scales_fp16);

                // Compute: accum * a_scale * b_scale
                __m512 scaled = _mm512_mul_ps(accum_f32, a_scales_f32);
                scaled = _mm512_mul_ps(scaled, b_scales_f32);

                // Horizontal reduction
                vectorized_result += _mm512_reduce_add_ps(scaled);
            }

            // Scalar tail
            for (; kb < K_blocks; ++kb)
            {
                vectorized_result += compute_per_block_contribution(accum_values[kb], a_scales[kb], b_scales[kb]);
            }

            // Compare with tolerance (FP32 rounding differences expected)
            float abs_diff = std::abs(scalar_result - vectorized_result);
            float rel_diff = abs_diff / std::max(std::abs(scalar_result), 1e-6f);

            return (rel_diff < 1e-5f); // 0.001% tolerance for FP32 rounding
        }

        /**
         * @brief Compute dot product using dpbusd (for testing)
         *
         * Simulates the dpbusd instruction to verify it computes what we expect.
         * dpbusd: dst[i] = src[i] + a.byte[4*i+0]*b.byte[4*i+0] + ... + a.byte[4*i+3]*b.byte[4*i+3]
         * where a is unsigned bytes, b is signed bytes
         */
        static inline int32_t compute_dpbusd_dot_product(const uint8_t *B_unsigned,
                                                         const int8_t *A_signed, int length)
        {
            int32_t sum = 0;
            for (int i = 0; i < length; ++i)
            {
                sum += static_cast<int32_t>(B_unsigned[i]) * static_cast<int32_t>(A_signed[i]);
            }
            return sum;
        }

        /**
         * @brief Compute reference dot product (signed × signed, no dpbusd)
         *
         * This is what the edge case microkernel computes.
         */
        static inline int32_t compute_reference_dot_product(const int8_t *B_signed,
                                                            const int8_t *A_signed, int length)
        {
            int32_t sum = 0;
            for (int i = 0; i < length; ++i)
            {
                sum += static_cast<int32_t>(B_signed[i]) * static_cast<int32_t>(A_signed[i]);
            }
            return sum;
        }

        /**
         * @brief Apply compensation to dpbusd result
         *
         * dpbusd computes Σ((B+128) × A), we want Σ(B × A)
         * So: Σ(B × A) = dpbusd_result - 128 × Σ(A)
         *
         * @param dpbusd_result Raw dpbusd accumulator value
         * @param sum_qs Sum of A quantized values: Σ(qs)
         * @return Compensated result (still in quantized space, needs scaling)
         */
        static inline int32_t apply_dpbusd_compensation(int32_t dpbusd_result, float sum_qs)
        {
            // CRITICAL: sum_qs is ALREADY Σ(A_quantized), not FP32!
            // Do NOT divide by scale here - that's done later
            float compensation = 128.0f * sum_qs;
            return static_cast<int32_t>(static_cast<float>(dpbusd_result) - compensation);
        }

        /**
         * @brief Verify dpbusd + compensation equals reference
         *
         * Tests that: dpbusd((B+128), A) - 128*Σ(A) = Σ(B × A)
         *
         * @param B_signed B values in signed int8 space
         * @param A_signed A values in signed int8 space
         * @param sum_a Q8_1 precomputed sum: s = d × Σ(A_qs) (FP32 space!)
         * @param a_scale Q8_1 scale factor: d
         * @param length Number of elements
         *
         * CRITICAL: In Q8_1 format, sum_a is the FP32-space precomputed sum (s),
         *           NOT the quantized-space sum (Σ(qs)). To get Σ(qs), divide by d.
         */
        static inline bool verify_dpbusd_compensation(const int8_t *B_signed, const int8_t *A_signed,
                                                      float sum_a, float a_scale, int length)
        {
            // Convert B to unsigned for dpbusd
            std::vector<uint8_t> B_unsigned(length);
            for (int i = 0; i < length; ++i)
            {
                B_unsigned[i] = static_cast<uint8_t>(B_signed[i] + 128);
            }

            // Compute dpbusd result
            int32_t dpbusd_result = compute_dpbusd_dot_product(B_unsigned.data(), A_signed, length);

            // CRITICAL: sum_a is FP32-space precomputed sum (s = d × Σ(qs))
            // To get quantized-space sum, divide by scale: Σ(qs) = s / d
            float sum_qs = sum_a / std::max(a_scale, 1e-10f);

            // Apply compensation
            int32_t compensated = apply_dpbusd_compensation(dpbusd_result, sum_qs);

            // Compute reference (what edge case computes)
            int32_t reference = compute_reference_dot_product(B_signed, A_signed, length);

            // They should match exactly (integer arithmetic)
            return compensated == reference;
        }

        /**
         * @brief Main GEMM entry point: C = A × B (IActivationTensor × Q8_0 → FP32)
         *
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A / rows in B (must be multiple of 32)
         * @param A Activation tensor (any type implementing IActivationTensor/IQ8_1Decodable)
         * @param B Q8_0 quantized weight matrix (K × N)
         * @param C Output FP32 matrix (M × N)
         * @param ldc Leading dimension of C
         *
         * @note K must be a multiple of BLOCK_SIZE (32)
         * @note C must be pre-allocated (M × N floats)
         * @note A is accessed via IQ8_1Decodable interface (on-the-fly quantization for FP32/FP16/BF16)
         */
        static void gemm(int M, int N, int K,
                         const IActivationTensor &A,
                         const Q8_0Tensor &B,
                         float *C, int ldc)
        {
            // Compile-time parameter validation
            static_assert(MR > 0, "MR must be positive");
            static_assert(NR > 0, "NR must be positive");
            static_assert(MR % 8 == 0, "MR must be a multiple of 8 for vectorization");
            static_assert(NR % 2 == 0, "NR must be even (packed in pairs)");
            static_assert(NR % 8 == 0, "NR must be a multiple of 8 for batched reduction");
            static_assert(BLOCK_SIZE == 32, "BLOCK_SIZE must be 32 (Q8_0/Q8_1 format)");
            static_assert(PREFETCH_A >= 0, "PREFETCH_A must be non-negative");
            static_assert(VECTOR_WIDTH == 16, "VECTOR_WIDTH must be 16 (AVX-512)");
            static_assert(JR_UNROLL >= 1 && JR_UNROLL <= 8, "JR_UNROLL must be in range [1, 8]");
            static_assert(JR_UNROLL == 1 || JR_UNROLL == 2 || JR_UNROLL == 4 ||
                              JR_UNROLL == 6 || JR_UNROLL == 8,
                          "JR_UNROLL must be 1, 2, 4, 6, or 8");
            static_assert(NR % JR_UNROLL == 0, "NR must be divisible by JR_UNROLL");

            // Validate dimensions
            if (K % BLOCK_SIZE != 0)
            {
                throw std::runtime_error("Q8_1 GEMM: K must be multiple of 32");
            }

            // Diagnostic: log thread count on first call
            static bool first_call = true;
            if (first_call)
            {
                int max_threads = omp_get_max_threads();
#pragma omp parallel
                {
#pragma omp single
                    {
                        int num_threads = omp_get_num_threads();
                        // std::cout << "[Q8_1 GEMM] Using " << num_threads << " threads (max=" << max_threads << ")" << std::endl;
                    }
                }
                first_call = false;
            }

            // Cast A to IQ8_1Decodable interface (required for block access)
            const IQ8_1Decodable *A_decodable = dynamic_cast<const IQ8_1Decodable *>(&A);
            if (!A_decodable)
            {
                throw std::runtime_error("Q8_1 GEMM: A must implement IQ8_1Decodable interface");
            }

            // Profiling: measure phase timings (enabled via env var, evaluated ONCE at startup)
            static const bool enable_profiling = (std::getenv("Q8_0_PROFILE") != nullptr);
            static const bool enable_detailed_profiling = (std::getenv("Q8_0_PROFILE_DETAILED") != nullptr);
            std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_pack_start, t_pack_end, t_gemm_start, t_gemm_end;
            if (enable_profiling)
            {
                t_start = std::chrono::high_resolution_clock::now();
            }

            const int K_blocks = K / BLOCK_SIZE;

            // Determine NC and KC blocking (auto-tune based on cache sizes)
            // NC: L2 cache blocking for N dimension (targets ~512KB of B data in L2)
            // KC: K-dimension blocking limited by storage (MAX_K_BLOCKS=128)
            const int nc_blocks = (NC == 0) ? compute_nc_blocking(N, K_blocks) : NC;
            const int kc_blocks = (KC == 0) ? compute_kc_blocking(K_blocks) : KC;

            // Get raw B block data (weights remain Q8_0)
            // B is structured as: [rows][K_blocks] with Q8_0Block structs
            const Q8_0Block *B_blocks = reinterpret_cast<const Q8_0Block *>(
                B.get_raw_block_at(0, 0));

            // NOTE: A blocks are accessed via IQ8_1Decodable interface (on-the-fly quantization)
            // No direct pointer access needed - we call A_decodable->decode_to_q8_1(row, k_block) per block

            // KC blocking loop (outermost): Process K in chunks for cache efficiency
            for (int kc_start = 0; kc_start < K_blocks; kc_start += kc_blocks)
            {
                const int kc_size = std::min(kc_blocks, K_blocks - kc_start);

                // NC blocking loop: Process N in chunks for L3 cache reuse
                for (int nc_start = 0; nc_start < N; nc_start += nc_blocks)
                {
                    const int nc_size = std::min(nc_blocks, N - nc_start);
                    const int nc_panels = (nc_size + NR - 1) / NR;

                    // Pack B into panels (one per NR columns) - SERIAL
                    // This is only ~2.6% of total work (802K ops vs 30M total)
                    // Serial execution avoids false sharing and is simpler
                    std::vector<PackedBPanel> B_panels;
                    B_panels.reserve(nc_panels);

                    if (enable_profiling && kc_start == 0 && nc_start == 0)
                    {
                        t_pack_start = std::chrono::high_resolution_clock::now();
                    }
                    for (int panel = 0; panel < nc_panels; ++panel)
                    {
                        const int j_start = nc_start + panel * NR;
                        const int panel_width = std::min(NR, N - j_start);

                        B_panels.emplace_back(kc_size);
                        pack_B_panel(panel_width, kc_size,
                                     B_blocks + j_start * K_blocks + kc_start,
                                     B_panels.back());
                    }
                    if (enable_profiling && kc_start == K_blocks - kc_size && nc_start == N - nc_size)
                    {
                        t_pack_end = std::chrono::high_resolution_clock::now();
                    }

                    // M-loop parallelization - this is where 97%+ of the work is
                    if (enable_profiling && kc_start == 0 && nc_start == 0)
                    {
                        t_gemm_start = std::chrono::high_resolution_clock::now();
                    }
#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < M; i += MR)
                    {
                        const int m_block = std::min(MR, M - i);

                        // N-loop (panels of NR columns)
                        for (int panel = 0; panel < nc_panels; ++panel)
                        {
                            const int j_start = nc_start + panel * NR;
                            const int n_block = std::min(NR, N - j_start);

                            float *C_block = C + i * ldc + j_start;

                            // Call microkernel
                            if (m_block == MR && n_block == NR)
                            {
                                // Fast path: full MR×NR tile
                                // Pass i (row index), kc_start (K-block offset), A_decodable interface
                                microkernel_full(kc_size, i, kc_start, A_decodable,
                                                 B_panels[panel], C_block, ldc);
                            }
                            else
                            {
                                // Edge case: partial tile
                                microkernel_edge(m_block, n_block, kc_size, i, kc_start, A_decodable,
                                                 B_panels[panel], C_block, ldc);
                            }
                        }
                    }
                    if (enable_profiling && kc_start == K_blocks - kc_size && nc_start == N - nc_size)
                    {
                        t_gemm_end = std::chrono::high_resolution_clock::now();
                    }
                } // NC loop
            } // KC loop

            // Print profiling results
            if (enable_profiling)
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                double t_total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                double t_pack_ms = std::chrono::duration<double, std::milli>(t_pack_end - t_pack_start).count();
                double t_gemm_ms = std::chrono::duration<double, std::milli>(t_gemm_end - t_gemm_start).count();

#pragma omp critical
                {
                    std::cout << "[Q8_1 PROFILE] M=" << M << " N=" << N << " K=" << K << std::endl;
                    std::cout << "  B packing:  " << std::fixed << std::setprecision(2)
                              << t_pack_ms << " ms (" << std::setprecision(1)
                              << (100.0 * t_pack_ms / t_total_ms) << "%)" << std::endl;
                    std::cout << "  GEMM:       " << std::fixed << std::setprecision(2)
                              << t_gemm_ms << " ms (" << std::setprecision(1)
                              << (100.0 * t_gemm_ms / t_total_ms) << "%)" << std::endl;
                    std::cout << "  Total:      " << std::fixed << std::setprecision(2)
                              << t_total_ms << " ms" << std::endl;
                    double gflops = (2.0 * M * N * K) / (t_total_ms * 1e6);
                    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                              << gflops << " GFLOPS" << std::endl;
                }
            }
        }

    private:
        /**
         * @brief Compute optimal NC blocking based on L2/L3 cache size
         *
         * Target: Keep NC × KC block of B in L2/L3 cache
         * L2: ~1MB per core, L3: ~32MB shared
         * For NC × KC: aim for ~512KB of B data per NC block
         *
         * Each column × K_blocks = K_blocks × 34 bytes/block
         * Target: NC × K_blocks × 34 ≈ 512KB → NC ≈ 512000 / (K_blocks × 34)
         *
         * @param N Total N dimension
         * @param K_blocks Total K dimension in blocks
         * @return NC blocking size (multiple of NR)
         */
        static int compute_nc_blocking(int N, int K_blocks)
        {
            // Target 512KB of B data per NC block
            constexpr int TARGET_B_SIZE = 512 * 1024;
            constexpr int BYTES_PER_BLOCK = 34; // Q8_0Block size

            int nc_target = TARGET_B_SIZE / (K_blocks * BYTES_PER_BLOCK);

            // Round down to multiple of NR
            nc_target = (nc_target / NR) * NR;

            // Ensure at least NR, at most N
            nc_target = std::max(NR, std::min(nc_target, N));

            return nc_target;
        }

        /**
         * @brief Compute optimal KC blocking based on storage limits
         *
         * Microkernel uses thread-local storage with MAX_K_BLOCKS=128
         * KC blocking allows processing K > 128 blocks in multiple passes
         *
         * @param K_blocks Total K dimension in blocks
         * @return KC blocking size (capped at MAX_K_BLOCKS)
         */
        static int compute_kc_blocking(int K_blocks)
        {
            constexpr int MAX_K_BLOCKS = 128; // From microkernel storage
            return std::min(MAX_K_BLOCKS, K_blocks);
        }

    public:
        // Make pack_B_panel public for unit testing
        /**
         * @brief Pack B panel into column-major layout with padding
         *
         * Input:  B_blocks[panel_width][K_blocks] (Q8_0Block structs)
         * Output: packed.quants[K_blocks][NR][64] (each column = 32 data + 32 padding)
         *         packed.scales[NR][K_blocks] (FP16, TRANSPOSED for sequential access)
         *
         * Layout enables direct ZMM loads: quants[kb * NR * 64 + jr * 64] loads column jr of block kb
         *
         * OPTIMIZATION: B scales stored as [jr][kb] instead of [kb][jr] for stride-1 vectorization
         */
        static void pack_B_panel(int panel_width, int K_blocks,
                                 const Q8_0Block *B_blocks,
                                 PackedBPanel &packed)
        {

            uint8_t *quants_base = packed.quants.data();
            uint16_t *scales_base = packed.scales.data();

            // Process each K-block
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                // Extract scales for this block - TRANSPOSED layout [jr][kb]
                for (int jr = 0; jr < panel_width; ++jr)
                {
                    const Q8_0Block &block = B_blocks[jr * K_blocks + kb];
                    scales_base[jr * K_blocks + kb] = block.d; // Transposed!
                }
                // Zero-fill scales for partial panels
                for (int jr = panel_width; jr < NR; ++jr)
                {
                    scales_base[jr * K_blocks + kb] = 0; // Zero scale
                }

                // Pack quantized values: [kb][jr_pair][64] where 64 bytes = 2 columns
                // Layout: [col0_bytes0..31][col1_bytes0..31] (no padding waste!)
                for (int jr_pair = 0; jr_pair < NR / 2; ++jr_pair)
                {
                    int jr0 = jr_pair * 2;     // First column in pair
                    int jr1 = jr_pair * 2 + 1; // Second column in pair

                    // Get ZMM base for this pair
                    uint8_t *zmm_base = quants_base + (kb * (NR / 2) + jr_pair) * 64;

                    // Pack first column (bytes 0..31)
                    if (jr0 < panel_width)
                    {
                        const Q8_0Block &block = B_blocks[jr0 * K_blocks + kb];
                        for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                        {
                            zmm_base[k_in] = static_cast<uint8_t>(block.qs[k_in] + 128);
                        }
                    }
                    // else: already initialized to 128 (zero)

                    // Pack second column (bytes 32..63)
                    if (jr1 < panel_width)
                    {
                        const Q8_0Block &block = B_blocks[jr1 * K_blocks + kb];
                        for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                        {
                            zmm_base[32 + k_in] = static_cast<uint8_t>(block.qs[k_in] + 128);
                        }
                    }
                    // else: already initialized to 128 (zero)
                }
            }
        }

        /**
         * @brief Full MR×NR microkernel (no edge cases)
         *
         * Computes C[MR×NR] += A[8×K] × B[K×8]
         *
         * Register allocation (AVX-512):
         * - zmm0-zmm3: MR×NR int32 accumulators (4 ZMM × 16 int32 = 64 accumulators)
         *              Layout: [zmm0=rows0-1,cols0-3] [zmm1=rows2-3,cols0-3]
         *                      [zmm2=rows4-5,cols0-3] [zmm3=rows6-7,cols0-3]
         * - zmm4-zmm11: 8 sum_A accumulators (for compensation, one per row)
         * - zmm12-zmm19: Loaded A vectors (32×int8 per row)
         * - zmm20-zmm27: Loaded B vectors (32×uint8 per column)
         * - zmm28: ones vector (for sum_A accumulation)
         * - zmm29-zmm31: Scratch
         *
         * Strategy:
         * 1. K-loop processes blocks of 32 elements
         * 2. For each block:
         *    a. Load A[8 rows][32 elements] → 8 ZMM registers
         *    b. Load B[8 cols][32 elements] → 8 ZMM registers
         *    c. Compute MR×NR outer product using dpbusd
         *    d. Accumulate sum_A (for compensation) using same dpbusd
         * 3. After K-loop:
         *    a. Compute compensation: sum_A × 128
         *    b. Subtract from int32 accumulator
         *    c. Convert to FP32
         *    d. Apply per-block scales
         *    e. Accumulate to C
         */
        /**
         * @brief Full MR×NR microkernel with sum_qs-based compensation (ORIGINAL)
         *
         * This is the current production implementation using sum_qs = sA/dA reconstruction.
         * K-loop computes sum_qs as INT16, post-processing applies compensation in quantized space.
         *
         * @param K_blocks Number of K blocks to process
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        __attribute__((always_inline)) static void microkernel_full_sumqs(
            int K_blocks,
            int i_base,   // Row offset in A matrix
            int kc_start, // K-block offset
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            // Thread-local profiling accumulators (only allocated if detailed profiling enabled)
            thread_local static struct
            {
                double t_buffer_init_ms = 0.0;
                double t_k_loop_load_a_ms = 0.0;
                double t_k_loop_load_b_ms = 0.0;
                double t_k_loop_compute_ms = 0.0;
                double t_postprocess_ms = 0.0;
                int64_t call_count = 0;
            } perf_stats;

            static const bool enable_detailed_profiling = (std::getenv("Q8_0_PROFILE_DETAILED") != nullptr);

            auto t_microkernel_start = std::chrono::high_resolution_clock::now();

            // CORRECTED: Store per-block accumulations (not summed across blocks)
            // accum[ir][jr][kb] = dot product for row ir, col jr, K-block kb
            // sum_qs[kb][ir] = Σ(qs[i]) as INT16 for row ir, K-block kb (extracted from Q8_1 FP16 sum)
            //                  OPTIMIZATION: Convert FP16→INT16 in K-loop (eliminates post-processing overhead)
            //                  LAYOUT: TRANSPOSED to [K_blocks][MR] for contiguous vector stores (eliminates scatter!)
            //                  STORAGE: INT16 (range [-4096, 4096]) - halves memory bandwidth vs INT32

            auto t_buffer_init_start = std::chrono::high_resolution_clock::now();

            // OPTIMIZATION: Dynamic heap allocation (was thread-local static, but caused stack overflow)
            // With large microkernels (48×48), thread-local static arrays exceeded thread stack limits
            // Heap allocation adds ~1μs overhead but prevents crashes
            // For small microkernels (8×8-16×16), this is still L1-cacheable

            // Allocate buffers dynamically based on actual K_blocks needed
            std::vector<int32_t> accum_vec(MR * NR * K_blocks, 0);
            std::vector<int16_t> sum_qs_vec(K_blocks * MR, 0); // INT16: TRANSPOSED layout [K_blocks][MR], halves bandwidth
            std::vector<uint16_t> a_scales_vec(MR * K_blocks);

            int32_t *accum_storage = accum_vec.data();
            int16_t *sum_qs_storage = sum_qs_vec.data();
            uint16_t *a_scales_storage = a_scales_vec.data();

            auto t_buffer_init_end = std::chrono::high_resolution_clock::now();

            if (enable_detailed_profiling)
            {
                perf_stats.t_buffer_init_ms += std::chrono::duration<double, std::milli>(t_buffer_init_end - t_buffer_init_start).count();
            }

            // OPTIMIZATION: Allocate fp32 B scale storage for fused conversion
            // Convert B scales from fp16→fp32 ONCE (eliminates post-processing conversions)
            // Note: NOT converting A scales - doing so in K-loop adds 60% overhead to Load A phase
            std::vector<float> b_scales_f32_vec(NR * K_blocks);
            float *b_scales_f32_storage = b_scales_f32_vec.data();

            // Helper lambdas for 3D indexing
            auto accum = [&](int ir, int jr, int kb) -> int32_t &
            {
                return accum_storage[ir * NR * K_blocks + jr * K_blocks + kb];
            };
            auto sum_qs = [&](int kb, int ir) -> int16_t &
            {
                // TRANSPOSED layout: [K_blocks][MR] for contiguous vector stores
                // INT16 storage: range [-4096, 4096] for 32 signed int8 values
                return sum_qs_storage[kb * MR + ir];
            };
            auto a_scales = [&](int ir, int kb) -> uint16_t &
            {
                return a_scales_storage[ir * K_blocks + kb];
            };
            auto b_scales_f32 = [&](int jr, int kb) -> float &
            {
                return b_scales_f32_storage[jr * K_blocks + kb];
            };

            const __m512i ones_u8 = _mm512_set1_epi8(1);
            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();

            // Timing accumulators for K-loop phases
            double t_load_a_accum = 0.0;
            double t_load_b_accum = 0.0;
            double t_compute_accum = 0.0;

            // K-loop over blocks
            // NOTE: K-loop unrolling (4×) doesn't help because we need per-block results
            // (each block has different scales, can't accumulate across blocks in registers)
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                auto t_load_a_start = std::chrono::high_resolution_clock::now();

                // OPTIMIZATION: Prefetch A blocks for next iteration
                // Prefetch A blocks ahead (4 blocks empirically optimal)
                // NOTE: Prefetching via IQ8_1Decodable is more complex (indirect access)
                // For Q8_1Tensor (zero-copy), we could prefetch the decoded pointer
                // For FP32/FP16/BF16 (on-the-fly quantization), prefetching is less useful
                // Skipping prefetch for now - measure impact later if needed

                // Load A blocks via IQ8_1Decodable interface (supports all activation types)
                __m512i a_vec[MR];

                // PHASE 1: Load all MR blocks and extract FP16 metadata
                // (Can't vectorize decode_to_q8_1 calls - may do on-the-fly quantization)
                uint16_t sum_a_fp16[MR];
                uint16_t a_scale_fp16[MR];

                // OPTIMIZATION: Unroll by 2 to issue parallel 256-bit loads (exploit ILP)
                // Modern CPUs have 2-3 load units and can execute multiple loads simultaneously
                {
                    int ir = 0;
                    for (; ir + 1 < MR; ir += 2)
                    {
                        // Decode both blocks
                        const Q8_1Block *a_block_ptr0 = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb);
                        const Q8_1Block *a_block_ptr1 = A_decodable->decode_to_q8_1(i_base + ir + 1, kc_start + kb);
                        const Q8_1Block &a_block0 = *a_block_ptr0;
                        const Q8_1Block &a_block1 = *a_block_ptr1;

                        // Extract FP16 metadata for both blocks
                        sum_a_fp16[ir] = a_block0.s;
                        sum_a_fp16[ir + 1] = a_block1.s;
                        a_scale_fp16[ir] = a_block0.d;
                        a_scale_fp16[ir + 1] = a_block1.d;
                        a_scales(ir, kb) = a_block0.d;
                        a_scales(ir + 1, kb) = a_block1.d;

                        // Load both qs arrays in parallel (2 independent 256-bit loads)
                        // CPU can issue these simultaneously to different load units
                        __m256i a_ymm0 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(a_block0.qs));
                        __m256i a_ymm1 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(a_block1.qs));

                        // Zero-extend to 512 bits (zero-latency casts)
                        a_vec[ir] = _mm512_castsi256_si512(a_ymm0);
                        a_vec[ir + 1] = _mm512_castsi256_si512(a_ymm1);
                    }

                    // Scalar tail (handle odd MR)
                    for (; ir < MR; ++ir)
                    {
                        // Decode to Q8_1 format (zero-copy for Q8_1Tensor, on-the-fly for FP32/FP16/BF16)
                        const Q8_1Block *a_block_ptr = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb);
                        const Q8_1Block &a_block = *a_block_ptr;

                        // Extract FP16 metadata to temporary arrays
                        sum_a_fp16[ir] = a_block.s;
                        a_scale_fp16[ir] = a_block.d;
                        a_scales(ir, kb) = a_block.d;

                        // Load 32 bytes (256 bits) and zero-extend to 512 bits
                        __m256i a_ymm = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(a_block.qs));
                        // Zero-extend to 512 bits (upper 256 bits = 0)
                        a_vec[ir] = _mm512_castsi256_si512(a_ymm);
                    }
                }

                // PHASE 2: VECTORIZED sum_qs extraction (process 16 rows at a time)
                // Formula: sum_qs = round(sum_a / max(a_scale, 1e-10))
                constexpr int VEC_WIDTH = 16; // AVX-512: 16 FP32 values
                int ir = 0;

                for (; ir + VEC_WIDTH <= MR; ir += VEC_WIDTH)
                {
                    // Load 16 sum_a values (FP16) and convert to FP32
                    __m256i sum_a_vec = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&sum_a_fp16[ir]));
                    __m512 sum_a_f32 = _mm512_cvtph_ps(sum_a_vec);

                    // Load 16 a_scale values (FP16) and convert to FP32
                    __m256i a_scale_vec = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&a_scale_fp16[ir]));
                    __m512 a_scale_f32 = _mm512_cvtph_ps(a_scale_vec);

                    // Clamp scales to avoid division by zero
                    __m512 epsilon = _mm512_set1_ps(1e-10f);
                    __m512 a_scale_safe = _mm512_max_ps(a_scale_f32, epsilon);

                    // Vectorized division: sum_qs = sum_a / a_scale
                    __m512 sum_qs_f32 = _mm512_div_ps(sum_a_f32, a_scale_safe);

                    // Round to nearest integer
                    __m512 sum_qs_rounded = _mm512_roundscale_ps(sum_qs_f32,
                                                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                    // Convert to INT32
                    __m512i sum_qs_i32 = _mm512_cvtps_epi32(sum_qs_rounded);

                    // Convert INT32 to INT16 (pack with saturation for safety)
                    // Range check: 32 signed int8 values sum to [-4096, 4096], well within INT16 range
                    __m256i sum_qs_i16 = _mm512_cvtsepi32_epi16(sum_qs_i32);

                    // Store to sum_qs array - TRANSPOSED layout enables contiguous vector store!
                    // sum_qs is [K_blocks, MR] (INT16), so sum_qs(kb, ir) through sum_qs(kb, ir+15) are consecutive
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&sum_qs(kb, ir)), sum_qs_i16);
                }

                // 8-wide vectorized tail (AVX2-compatible)
                for (; ir + 8 <= MR; ir += 8)
                {
                    // Load 8 sum_a values (FP16) and convert to FP32
                    __m128i sum_a_vec = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(&sum_a_fp16[ir]));
                    __m256 sum_a_f32 = _mm256_cvtph_ps(sum_a_vec);

                    // Load 8 a_scale values (FP16) and convert to FP32
                    __m128i a_scale_vec = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(&a_scale_fp16[ir]));
                    __m256 a_scale_f32 = _mm256_cvtph_ps(a_scale_vec);

                    // Clamp scales to avoid division by zero
                    __m256 epsilon = _mm256_set1_ps(1e-10f);
                    __m256 a_scale_safe = _mm256_max_ps(a_scale_f32, epsilon);

                    // Vectorized division: sum_qs = sum_a / a_scale
                    __m256 sum_qs_f32 = _mm256_div_ps(sum_a_f32, a_scale_safe);

                    // Round to nearest integer
                    __m256 sum_qs_rounded = _mm256_round_ps(sum_qs_f32,
                                                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                    // Convert to INT32
                    __m256i sum_qs_i32 = _mm256_cvtps_epi32(sum_qs_rounded);

                    // Convert INT32 to INT16 (pack with saturation)
                    // AVX2: Pack two __m256i (8 int32 each) into one __m256i (16 int16)
                    // We only have 8 values, so zero the upper half
                    __m256i zero = _mm256_setzero_si256();
                    __m256i sum_qs_i16 = _mm256_packs_epi32(sum_qs_i32, zero); // [8 int16, 8 zeros]

                    // Extract lower 128 bits (8 int16 values) and store
                    __m128i sum_qs_i16_lo = _mm256_castsi256_si128(sum_qs_i16);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(&sum_qs(kb, ir)), sum_qs_i16_lo);
                }

                // 4-wide vectorized tail (AVX2-compatible)
                for (; ir + 4 <= MR; ir += 4)
                {
                    // Load 4 sum_a values (FP16) and convert to FP32
                    __m128i sum_a_vec = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(&sum_a_fp16[ir]));
                    __m128 sum_a_f32 = _mm_cvtph_ps(sum_a_vec);

                    // Load 4 a_scale values (FP16) and convert to FP32
                    __m128i a_scale_vec = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(&a_scale_fp16[ir]));
                    __m128 a_scale_f32 = _mm_cvtph_ps(a_scale_vec);

                    // Clamp scales to avoid division by zero
                    __m128 epsilon = _mm_set1_ps(1e-10f);
                    __m128 a_scale_safe = _mm_max_ps(a_scale_f32, epsilon);

                    // Vectorized division: sum_qs = sum_a / a_scale
                    __m128 sum_qs_f32 = _mm_div_ps(sum_a_f32, a_scale_safe);

                    // Round to nearest integer
                    __m128 sum_qs_rounded = _mm_round_ps(sum_qs_f32,
                                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                    // Convert to INT32
                    __m128i sum_qs_i32 = _mm_cvtps_epi32(sum_qs_rounded);

                    // Convert INT32 to INT16 (pack with saturation)
                    // SSE: Pack two __m128i (4 int32 each) into one __m128i (8 int16)
                    __m128i zero = _mm_setzero_si128();
                    __m128i sum_qs_i16 = _mm_packs_epi32(sum_qs_i32, zero); // [4 int16, 4 zeros]

                    // Store lower 64 bits (4 int16 values)
                    _mm_storel_epi64(reinterpret_cast<__m128i *>(&sum_qs(kb, ir)), sum_qs_i16);
                }

                // Scalar tail (if MR not multiple of 4)
                for (; ir < MR; ++ir)
                {
                    float sum_a_fp32 = fp16_to_fp32(sum_a_fp16[ir]);
                    float a_scale_fp32 = fp16_to_fp32(a_scale_fp16[ir]);
                    float sum_qs_fp32 = sum_a_fp32 / std::max(a_scale_fp32, 1e-10f);
                    int32_t sum_qs_i32 = static_cast<int32_t>(std::round(sum_qs_fp32));

// Range check in debug builds (32 signed int8: [-128,127] × 32 = [-4096, 4096])
#ifndef NDEBUG
                    if (sum_qs_i32 < -4096 || sum_qs_i32 > 4096)
                    {
                        std::cerr << "WARNING: sum_qs out of range: " << sum_qs_i32 << " at ir=" << ir << ", kb=" << kb << std::endl;
                    }
#endif

                    sum_qs(kb, ir) = static_cast<int16_t>(sum_qs_i32);
                }

                auto t_load_a_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_load_a_accum += std::chrono::duration<double, std::milli>(t_load_a_end - t_load_a_start).count();
                }

                auto t_load_b_start = std::chrono::high_resolution_clock::now();

                // Load B blocks for NR columns - OPTIMIZED: Parallel 2× 256-bit loads (exploit dual load ports!)
                // Layout: [kb][jr_pair][64] where 64 bytes = [col0_32bytes][col1_32bytes]
                // Instead of loading 512 bits then extracting, issue 2 parallel 256-bit loads
                __m512i b_vec[NR];
                const uint8_t *B_block_base = B_quants + kb * (NR / 2) * 64;

                for (int jr_pair = 0; jr_pair < NR / 2; ++jr_pair)
                {
                    // Base pointer for this column pair
                    const uint8_t *pair_base = B_block_base + jr_pair * 64;

                    // PARALLEL LOADS: CPU issues both simultaneously to AGU0/AGU1!
                    // Load first column (bytes 0-31, independent address)
                    __m256i col0_ymm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(pair_base));

                    // Load second column (bytes 32-63, independent address)
                    __m256i col1_ymm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(pair_base + 32));

                    // Zero-extend to 512 bits (zero-latency casts)
                    b_vec[jr_pair * 2] = _mm512_castsi256_si512(col0_ymm);
                    b_vec[jr_pair * 2 + 1] = _mm512_castsi256_si512(col1_ymm);
                }

                auto t_load_b_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_load_b_accum += std::chrono::duration<double, std::milli>(t_load_b_end - t_load_b_start).count();
                }

                auto t_compute_start = std::chrono::high_resolution_clock::now();

                // Compute MR×NR outer product FOR THIS KB
                // NOTE: sum_A NO LONGER COMPUTED - use pre-computed Q8_1 sums instead!
                // Q8_1 format: s = d × Σ(qs[i]), so Σ(qs[i]) = s / d
                // OPTIMIZATION: Process 2 columns at once to exploit ILP (2× unrolling - sweet spot)
                for (int ir = 0; ir < MR; ++ir)
                {
                    int jr = 0;
                    // Process 2 columns per iteration (exploit ILP - 2 dpbusd can execute in parallel)
                    for (; jr + 1 < NR; jr += 2)
                    {
                        // Issue 2 dpbusd operations before any reductions
                        __m512i acc_vec0 = _mm512_setzero_si512();
                        __m512i acc_vec1 = _mm512_setzero_si512();

                        acc_vec0 = _mm512_dpbusd_epi32(acc_vec0, b_vec[jr], a_vec[ir]);
                        acc_vec1 = _mm512_dpbusd_epi32(acc_vec1, b_vec[jr + 1], a_vec[ir]);

                        // Parallel horizontal reductions
                        int32_t dot0 = _mm512_reduce_add_epi32(acc_vec0);
                        int32_t dot1 = _mm512_reduce_add_epi32(acc_vec1);

                        accum(ir, jr, kb) = dot0;
                        accum(ir, jr + 1, kb) = dot1;
                    }

                    // Scalar tail for odd NR
                    for (; jr < NR; ++jr)
                    {
                        __m512i acc_vec = _mm512_setzero_si512();
                        acc_vec = _mm512_dpbusd_epi32(acc_vec, b_vec[jr], a_vec[ir]);

                        int32_t dot = _mm512_reduce_add_epi32(acc_vec);
                        accum(ir, jr, kb) = dot;
                    }
                }

                auto t_compute_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_compute_accum += std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
                }
            }

            // Accumulate K-loop phase timings
            if (enable_detailed_profiling)
            {
                perf_stats.t_k_loop_load_a_ms += t_load_a_accum;
                perf_stats.t_k_loop_load_b_ms += t_load_b_accum;
                perf_stats.t_k_loop_compute_ms += t_compute_accum;
            }

            // FUSED CONVERSION: Convert B scales fp16→fp32 (vectorized, eliminates post-processing conversions)
            // B scales layout: [jr][kb] (transposed for post-processing)
            // B_scales already declared above, use B_K_blocks for correct dimension
            const int B_K_blocks = B_packed.K_blocks;

            for (int jr = 0; jr < NR; ++jr)
            {
                int kb = 0;
                // AVX-512 16-wide vectorized conversion
                for (; kb + 16 <= K_blocks; kb += 16)
                {
                    __m256i scales_fp16 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m512 scales_fp32 = _mm512_cvtph_ps(scales_fp16);
                    _mm512_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // AVX2 8-wide vectorized tail
                for (; kb + 8 <= K_blocks; kb += 8)
                {
                    __m128i scales_fp16 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m256 scales_fp32 = _mm256_cvtph_ps(scales_fp16);
                    _mm256_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // AVX2 4-wide vectorized tail
                for (; kb + 4 <= K_blocks; kb += 4)
                {
                    __m128i scales_fp16 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m128 scales_fp32 = _mm_cvtph_ps(scales_fp16);
                    _mm_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // Scalar tail (only 0-3 blocks remaining)
                for (; kb < K_blocks; ++kb)
                {
                    b_scales_f32(jr, kb) = fp16_to_fp32(B_scales[jr * B_K_blocks + kb]);
                }
            }

            auto t_postprocess_start = std::chrono::high_resolution_clock::now();

            // Post-processing: OPTIMIZED with batched reductions and B scale pre-conversion
            // Key optimizations:
            // 1. FUSED B CONVERSIONS: B scales already converted to fp32 once (eliminates NR cvtph_ps calls)
            // 2. VECTORIZED A CONVERSIONS: Convert VECTOR_WIDTH A scales per iteration (amortized cost)
            // 3. BATCHED REDUCTIONS: Process jr in chunks of 8, reducing 1024→128 horizontal sums
            //
            // Formula: C[i,j] = Σ_kb ((accum[kb] - 128*sum_qs[kb]) * a_scale[kb] * b_scale[kb])
            // NOTE: dpbusd computes Σ((B+128)*A), so we must subtract 128*Σ(A) to get Σ(B*A)
            //       where Σ(A) = sum_qs = sum_a / a_scale (from Q8_1 pre-computed sum)

            // Process ir rows sequentially (cache-friendly for accum access)
            for (int ir = 0; ir < MR; ++ir)
            {
                // BATCHED REDUCTION: Process jr columns in chunks of 8
                // Note: NR must be a multiple of JR_BATCH
                constexpr int JR_BATCH = 8;
                static_assert(NR % JR_BATCH == 0, "NR must be a multiple of JR_BATCH");

                for (int jr_base = 0; jr_base < NR; jr_base += JR_BATCH)
                {
                    // Accumulate results for 8 columns simultaneously
                    __m512 result_vecs[JR_BATCH];
                    for (int jj = 0; jj < JR_BATCH; ++jj)
                    {
                        result_vecs[jj] = _mm512_setzero_ps();
                    }

                    // Process K_blocks in chunks of VECTOR_WIDTH
                    int kb = 0;
                    for (; kb + VECTOR_WIDTH <= K_blocks; kb += VECTOR_WIDTH)
                    {
                        // Load VECTOR_WIDTH sum_qs values (INT16) and convert to INT32, then FP32
                        // OPTIMIZATION: sum_qs stored as INT16 in K-loop (halves memory bandwidth!)
                        // NOTE: With transposed layout [K_blocks][MR], we need to gather across K-blocks for same row
                        // AVX-512 has no native int16 gather, so we use two int32 gathers with scale=2
                        __m256i gather_indices_256 = _mm256_setr_epi32(
                            0 * MR, 1 * MR, 2 * MR, 3 * MR, 4 * MR, 5 * MR, 6 * MR, 7 * MR);

                        // Gather first 8 int32 values (each contains 2 int16s, we want lower 16 bits)
                        __m256i sum_qs_i32_lo_raw = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices_256, 2);
                        __m256i sum_qs_i32_lo = _mm256_srai_epi32(_mm256_slli_epi32(sum_qs_i32_lo_raw, 16), 16);

                        // Gather second 8 int32 values
                        __m256i sum_qs_i32_hi_raw = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb + 8, ir)), gather_indices_256, 2);
                        __m256i sum_qs_i32_hi = _mm256_srai_epi32(_mm256_slli_epi32(sum_qs_i32_hi_raw, 16), 16);

                        // Combine into 512-bit (16 int32 values)
                        __m512i sum_qs_i32 = _mm512_inserti64x4(_mm512_castsi256_si512(sum_qs_i32_lo), sum_qs_i32_hi, 1);
                        __m512 sum_qs_f32 = _mm512_cvtepi32_ps(sum_qs_i32);

                        // Convert VECTOR_WIDTH A scales fp16→fp32 (vectorized, amortized across JR_BATCH columns)
                        __m256i a_scales_fp16 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(&a_scales(ir, kb)));
                        __m512 a_scales_vec = _mm512_cvtph_ps(a_scales_fp16);

                        // Precompute compensation constant
                        const __m512 compensation_const = _mm512_set1_ps(128.0f);

                        // Process all 8 columns in the batch
                        for (int jj = 0; jj < JR_BATCH; ++jj)
                        {
                            int jr = jr_base + jj;

                            // Load VECTOR_WIDTH int32 accumulators and convert to float
                            __m512i accum_i32 = _mm512_loadu_si512(&accum(ir, jr, kb));
                            __m512 accum_f32 = _mm512_cvtepi32_ps(accum_i32);

                            // Load VECTOR_WIDTH fp32 b_scales (already converted!)
                            __m512 b_scales_vec = _mm512_loadu_ps(&b_scales_f32(jr, kb));

                            // COMPENSATION: compensated = accum - sum_qs * 128 (FUSED FMA!)
                            // Now matches Q8_0 exactly (no FP16 conversion, no division, uses fused FMA)
                            __m512 compensated = _mm512_fnmadd_ps(sum_qs_f32, compensation_const, accum_f32); // 4 cycles

                            // Compute: compensated * a_scale * b_scale
                            __m512 scaled = _mm512_mul_ps(compensated, a_scales_vec);
                            scaled = _mm512_mul_ps(scaled, b_scales_vec);

                            // Accumulate to result
                            result_vecs[jj] = _mm512_add_ps(result_vecs[jj], scaled);
                        }
                    }

                    // Horizontal reductions for the batch (128 reductions instead of 1024!)
                    float results[JR_BATCH];
                    for (int jj = 0; jj < JR_BATCH; ++jj)
                    {
                        results[jj] = _mm512_reduce_add_ps(result_vecs[jj]);
                    }

                    // VECTORIZED TAIL: Process remaining K_blocks in chunks of 8
                    constexpr int TAIL_VEC_WIDTH = 8;
                    for (; kb + TAIL_VEC_WIDTH <= K_blocks; kb += TAIL_VEC_WIDTH)
                    {
                        // Load 8 sum_qs values (INT16) and convert to INT32, then FP32
                        // NOTE: With transposed layout, use AVX2 gather with scale=2 for int16
                        __m256i gather_indices = _mm256_setr_epi32(
                            0 * MR, 1 * MR, 2 * MR, 3 * MR, 4 * MR, 5 * MR, 6 * MR, 7 * MR);
                        // Gather 8 int32 values (each contains 2 int16s, we want lower 16 bits)
                        __m256i sum_qs_i32_raw = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices, 2);
                        // Extract lower 16 bits and sign-extend to 32-bit
                        __m256i sum_qs_i32 = _mm256_srai_epi32(_mm256_slli_epi32(sum_qs_i32_raw, 16), 16);
                        __m256 sum_qs_f32 = _mm256_cvtepi32_ps(sum_qs_i32);

                        // Load 8 A scales (FP16) and convert to FP32
                        __m128i a_scales_fp16 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(&a_scales(ir, kb)));
                        __m256 a_scales_vec = _mm256_cvtph_ps(a_scales_fp16);

                        // Precompute compensation constant
                        const __m256 compensation_const = _mm256_set1_ps(128.0f);

                        // Process all 8 columns in the batch
                        for (int jj = 0; jj < JR_BATCH; ++jj)
                        {
                            int jr = jr_base + jj;

                            // Load 8 int32 accumulators and convert to float
                            __m256i accum_i32 = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i *>(&accum(ir, jr, kb)));
                            __m256 accum_f32 = _mm256_cvtepi32_ps(accum_i32);

                            // Load 8 fp32 b_scales (already converted!)
                            __m256 b_scales_vec = _mm256_loadu_ps(&b_scales_f32(jr, kb));

                            // COMPENSATION: compensated = accum - sum_qs * 128 (FUSED FMA!)
                            __m256 compensated = _mm256_fnmadd_ps(sum_qs_f32, compensation_const, accum_f32);

                            // Compute: compensated * a_scale * b_scale
                            __m256 scaled = _mm256_mul_ps(compensated, a_scales_vec);
                            scaled = _mm256_mul_ps(scaled, b_scales_vec);

                            // Horizontal reduction (AVX2-compatible)
                            // Sum 8 floats: [a b c d e f g h] → a+b+c+d+e+f+g+h
                            __m128 lo = _mm256_castps256_ps128(scaled);
                            __m128 hi = _mm256_extractf128_ps(scaled, 1);
                            __m128 sum4 = _mm_add_ps(lo, hi);     // [a+e, b+f, c+g, d+h]
                            __m128 shuf = _mm_movehdup_ps(sum4);  // [b+f, b+f, d+h, d+h]
                            __m128 sum2 = _mm_add_ps(sum4, shuf); // [a+b+e+f, *, c+d+g+h, *]
                            shuf = _mm_movehl_ps(shuf, sum2);     // [c+d+g+h, *, *, *]
                            __m128 sum1 = _mm_add_ss(sum2, shuf); // [a+b+c+d+e+f+g+h, *, *, *]
                            results[jj] += _mm_cvtss_f32(sum1);
                        }
                    }

                    // AVX2 4-wide vectorized tail (process 4 K_blocks at once)
                    constexpr int TAIL_VEC_WIDTH_4 = 4;

                    for (; kb + TAIL_VEC_WIDTH_4 <= K_blocks; kb += TAIL_VEC_WIDTH_4)
                    {
                        // Load 4 sum_qs (INT16) and convert to FP32
                        // NOTE: With transposed layout, use AVX gather intrinsic
                        // Indices: [0*MR, 1*MR, 2*MR, 3*MR] in int16 (multiply by 2 for byte offset)
                        __m128i gather_indices = _mm_setr_epi32(0 * MR, 1 * MR, 2 * MR, 3 * MR);
                        // AVX gather for 4 int32 values (each contains 2 int16s, we want lower 16 bits)
                        __m128i sum_qs_i32_raw = _mm_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices, 2);
                        // Extract lower 16 bits and sign-extend to 32-bit
                        __m128i sum_qs_i32 = _mm_srai_epi32(_mm_slli_epi32(sum_qs_i32_raw, 16), 16);
                        __m128 sum_qs_f32 = _mm_cvtepi32_ps(sum_qs_i32);

                        // Load 4 A scales (FP16) and convert to FP32
                        // Note: Only loading 64 bits (4×FP16), so use _mm_loadl_epi64
                        __m128i a_scales_fp16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&a_scales(ir, kb)));
                        __m128 a_scales_vec = _mm_cvtph_ps(a_scales_fp16);

                        for (int jj = 0; jj < JR_BATCH; ++jj)
                        {
                            int jr = jr_base + jj;

                            // Load 4 accumulators (INT32) and convert to FP32
                            __m128i accum_i32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&accum(ir, jr, kb)));
                            __m128 accum_f32 = _mm_cvtepi32_ps(accum_i32);

                            // Load 4 B scales (FP32, already converted!)
                            __m128 b_scales_vec = _mm_loadu_ps(&b_scales_f32(jr, kb));

                            // Compensation: compensated = accum - sum_qs * 128
                            __m128 compensation_const = _mm_set1_ps(128.0f);
                            __m128 compensated = _mm_fnmadd_ps(sum_qs_f32, compensation_const, accum_f32);

                            // Scale: result = compensated * a_scale * b_scale
                            __m128 scaled = _mm_mul_ps(compensated, a_scales_vec);
                            scaled = _mm_mul_ps(scaled, b_scales_vec);

                            // Horizontal reduction: 4→1
                            __m128 shuf = _mm_movehdup_ps(scaled);  // [b, b, d, d]
                            __m128 sum2 = _mm_add_ps(scaled, shuf); // [a+b, *, c+d, *]
                            shuf = _mm_movehl_ps(shuf, sum2);       // [c+d, *, *, *]
                            __m128 sum1 = _mm_add_ss(sum2, shuf);   // [a+b+c+d, *, *, *]
                            results[jj] += _mm_cvtss_f32(sum1);
                        }
                    }

                    // Scalar tail for remaining K_blocks (< 4)
                    for (; kb < K_blocks; ++kb)
                    {
                        // Load pre-extracted sum_qs (INT32) and A scale (FP16)
                        // OPTIMIZATION: sum_qs pre-extracted as INT32 in K-loop (no FP16 conversion, no division!)
                        // Load sum_qs (INT16), convert to INT32 for computation
                        int16_t sum_qs_val_i16 = sum_qs(kb, ir);
                        int32_t sum_qs_val = static_cast<int32_t>(sum_qs_val_i16);
                        float a_scale = fp16_to_fp32(a_scales(ir, kb));

                        for (int jj = 0; jj < JR_BATCH; ++jj)
                        {
                            int jr = jr_base + jj;
                            float b_scale = b_scales_f32(jr, kb); // Already fp32!

                            // COMPENSATION: compensated = accum - sum_qs * 128 (now matches Q8_0!)
                            // dpbusd gives Σ((B+128)*A), we want Σ(B*A) = Σ((B+128)*A) - 128*Σ(A)
                            int32_t accum_val = accum(ir, jr, kb);
                            float compensated = static_cast<float>(accum_val - sum_qs_val * 128);

                            results[jj] += compensated * a_scale * b_scale;
                        }
                    }

                    // Write results to C
                    for (int jj = 0; jj < JR_BATCH; ++jj)
                    {
                        int jr = jr_base + jj;
                        C[ir * ldc + jr] = results[jj];
                    }
                }
            }

            auto t_postprocess_end = std::chrono::high_resolution_clock::now();

            // Update profiling statistics
            if (enable_detailed_profiling)
            {
                perf_stats.t_postprocess_ms += std::chrono::duration<double, std::milli>(t_postprocess_end - t_postprocess_start).count();
                perf_stats.call_count++;

                // Print statistics every 1000 calls (per thread)
                if (perf_stats.call_count % 1000 == 0)
                {
                    int thread_id = omp_get_thread_num();
                    double total_ms = perf_stats.t_buffer_init_ms + perf_stats.t_k_loop_load_a_ms +
                                      perf_stats.t_k_loop_load_b_ms + perf_stats.t_k_loop_compute_ms +
                                      perf_stats.t_postprocess_ms;

#pragma omp critical
                    {
                        std::cout << "\n[Q8_1 DETAILED PROFILE - Thread " << thread_id << "]" << std::endl;
                        std::cout << "  Calls:              " << perf_stats.call_count << std::endl;
                        std::cout << "  Total time:         " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
                        std::cout << "  Avg per call:       " << std::fixed << std::setprecision(4) << (total_ms / perf_stats.call_count) << " ms" << std::endl;
                        std::cout << "\n  Phase Breakdown:" << std::endl;
                        std::cout << "    Buffer init:      " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_buffer_init_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_buffer_init_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    K-loop Load A:    " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_k_loop_load_a_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_k_loop_load_a_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    K-loop Load B:    " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_k_loop_load_b_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_k_loop_load_b_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    K-loop Compute:   " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_k_loop_compute_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_k_loop_compute_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    Post-process:     " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_postprocess_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_postprocess_ms / total_ms) << "%)" << std::endl;

                        // Per-call averages for each phase
                        std::cout << "\n  Per-Call Averages:" << std::endl;
                        std::cout << "    Buffer init:      " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_buffer_init_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    K-loop Load A:    " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_k_loop_load_a_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    K-loop Load B:    " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_k_loop_load_b_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    K-loop Compute:   " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_k_loop_compute_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    Post-process:     " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_postprocess_ms / perf_stats.call_count) << " ms/call" << std::endl;
                    }
                }
            }
        } // End microkernel_full_sumqs

        /**
         * @brief Helper: Process N_COLS columns with dpbusd + compensation (compile-time unrolled)
         *
         * This helper eliminates code duplication by template-generating the unrolled loops.
         * All unroll factors (1, 2, 4, 6, 8) use the same implementation with compile-time bounds.
         *
         * @tparam N_COLS Number of columns to process (1, 2, 4, 6, or 8)
         * @tparam ScaleAccessor Type of the lambda for accessing B scales (deduced)
         */
        template <int N_COLS, typename ScaleAccessor>
        __attribute__((always_inline)) static void process_n_columns(
            int jr, int ir, int kb, int ldc,
            float sA, float dA,
            const __m512i *a_vec,
            const __m512i *b_vec,
            const ScaleAccessor &b_scales_f32,
            float *C)
        {
            // Allocate accumulators (compiler will optimize unused ones away)
            __m512i acc_vec[8];
            int32_t dp[8];
            float dB[8];
            float main[8];
            float comp[8];

// Issue all dpbusd operations first (maximize ILP)
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                acc_vec[i] = _mm512_setzero_si512();
                acc_vec[i] = _mm512_dpbusd_epi32(acc_vec[i], b_vec[jr + i], a_vec[ir]);
            }

// Horizontal reductions (can execute in parallel)
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                dp[i] = _mm512_reduce_add_epi32(acc_vec[i]);
            }

// Load B scales and compute compensation
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                dB[i] = b_scales_f32(jr + i, kb);
                main[i] = static_cast<float>(dp[i]) * dA * dB[i];
                comp[i] = 128.0f * sA * dB[i];
            }

// Write results to C
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                C[ir * ldc + jr + i] += main[i] - comp[i];
            }
        }

        /**
         * @brief Full MR×NR microkernel with sA-based compensation (EXPERIMENTAL)
         *
         * This is an experimental implementation using direct sA (dA * sum_qs) metadata.
         * K-loop stores sA as FP16, post-processing applies compensation in real space.
         *
         * Mathematical equivalence:
         *   OLD: result = (accum - 128*sum_qs) * dA * dB  where sum_qs = sA/dA
         *   NEW: result = accum*dA*dB - 128*sA*dB
         *
         * Expected benefits:
         *   - Eliminates division (sA/dA) from K-loop
         *   - Eliminates int16 gather (4-5 cycle latency) → aligned FP16 load (1 cycle)
         *   - Better numerical properties (no FP→int→FP conversion chain)
         *   - Estimated +8-15% performance improvement
         *
         * @param K_blocks Number of K blocks to process
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        __attribute__((always_inline)) static void microkernel_full_sa(
            int K_blocks,
            int i_base,
            int kc_start,
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            // ===== STREAMING APPROACH: No accum_vec! =====
            // Only allocate metadata buffers (sA, dA for A; dB_f32 for B)
            // This saves 2MB of memory traffic (MR * NR * K_blocks * 4 bytes)
            std::vector<uint16_t> a_sums_vec(MR * K_blocks); // Store sA directly as FP16
            std::vector<uint16_t> a_scales_vec(MR * K_blocks);
            std::vector<float> b_scales_f32_vec(NR * K_blocks);

            uint16_t *a_sums_storage = a_sums_vec.data();
            uint16_t *a_scales_storage = a_scales_vec.data();
            float *b_scales_f32_storage = b_scales_f32_vec.data();

            // Helper lambdas for 2D indexing (no accum!)
            auto a_sums = [&](int ir, int kb) -> uint16_t &
            {
                return a_sums_storage[ir * K_blocks + kb];
            };
            auto a_scales = [&](int ir, int kb) -> uint16_t &
            {
                return a_scales_storage[ir * K_blocks + kb];
            };
            auto b_scales_f32 = [&](int jr, int kb) -> float &
            {
                return b_scales_f32_storage[jr * K_blocks + kb];
            };

            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();
            const int B_K_blocks = B_packed.K_blocks;

            // ===== OPTIMIZED K-LOOP: Load A/B once per kb, immediately accumulate to C =====
            // Key insight from GPT 5.1: Keep original K-loop structure (good data reuse),
            // but replace "store to accum_vec" with immediate compensation and C accumulation.
            //
            // Benefits:
            //   - No 2MB accum_vec buffer
            //   - A blocks decoded ONCE per (ir, kb) - not NR/JR_BATCH times
            //   - B blocks loaded ONCE per kb - not MR times
            //   - Maintains dpbusd ILP (2-way unrolling)
            //   - Compensation applied immediately after dpbusd

            // Pre-convert B scales from FP16 to FP32 (vectorized, one-time cost)
            for (int jr = 0; jr < NR; ++jr)
            {
                int kb = 0;
                // AVX-512 16-wide vectorized conversion
                for (; kb + 16 <= K_blocks; kb += 16)
                {
                    __m256i scales_fp16 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m512 scales_fp32 = _mm512_cvtph_ps(scales_fp16);
                    _mm512_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // AVX2 8-wide vectorized tail
                for (; kb + 8 <= K_blocks; kb += 8)
                {
                    __m128i scales_fp16 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m256 scales_fp32 = _mm256_cvtph_ps(scales_fp16);
                    _mm256_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // SSE 4-wide vectorized tail
                for (; kb + 4 <= K_blocks; kb += 4)
                {
                    __m128i scales_fp16 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m128 scales_fp32 = _mm_cvtph_ps(scales_fp16);
                    _mm_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // Scalar tail (0-3 blocks)
                for (; kb < K_blocks; ++kb)
                {
                    b_scales_f32(jr, kb) = fp16_to_fp32(B_scales[jr * B_K_blocks + kb]);
                }
            }

            // Main K-loop: Process one K-block at a time
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                // PHASE 1: Decode A blocks ONCE per (ir, kb) and keep in a_vec[MR]
                __m512i a_vec[MR];
                for (int ir = 0; ir < MR; ++ir)
                {
                    const Q8_1Block *a_block_ptr = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb);
                    const Q8_1Block &a_block = *a_block_ptr;

                    // Store FP16 metadata (sA, dA) for later use
                    a_scales(ir, kb) = a_block.d;
                    a_sums(ir, kb) = a_block.s;

                    // Load qs[] into a_vec[ir] (reused across all NR columns)
                    __m256i a_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block.qs));
                    a_vec[ir] = _mm512_castsi256_si512(a_ymm);
                }

                // PHASE 2: Load B blocks ONCE per kb into b_vec[NR]
                __m512i b_vec[NR];
                const uint8_t *B_block_base = B_quants + kb * (NR / 2) * 64;

                for (int jr_pair = 0; jr_pair < NR / 2; ++jr_pair)
                {
                    const uint8_t *pair_base = B_block_base + jr_pair * 64;
                    __m256i col0_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pair_base));
                    __m256i col1_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pair_base + 32));
                    b_vec[jr_pair * 2] = _mm512_castsi256_si512(col0_ymm);
                    b_vec[jr_pair * 2 + 1] = _mm512_castsi256_si512(col1_ymm);
                }

                // PHASE 3: dpbusd for all MR×NR and immediately accumulate into C
                for (int ir = 0; ir < MR; ++ir)
                {
                    // Load A metadata for this (ir, kb) ONCE
                    float sA = fp16_to_fp32(a_sums(ir, kb));
                    float dA = fp16_to_fp32(a_scales(ir, kb));

                    // Process JR_UNROLL columns at a time, tail loop handles remainder
                    int jr = 0;
                    for (; jr + JR_UNROLL <= NR; jr += JR_UNROLL)
                    {
                        if constexpr (JR_UNROLL == 8)
                            process_n_columns<8>(jr, ir, kb, ldc, sA, dA, a_vec, b_vec, b_scales_f32, C);
                        else if constexpr (JR_UNROLL == 6)
                            process_n_columns<6>(jr, ir, kb, ldc, sA, dA, a_vec, b_vec, b_scales_f32, C);
                        else if constexpr (JR_UNROLL == 4)
                            process_n_columns<4>(jr, ir, kb, ldc, sA, dA, a_vec, b_vec, b_scales_f32, C);
                        else if constexpr (JR_UNROLL == 2)
                            process_n_columns<2>(jr, ir, kb, ldc, sA, dA, a_vec, b_vec, b_scales_f32, C);
                        else
                            process_n_columns<1>(jr, ir, kb, ldc, sA, dA, a_vec, b_vec, b_scales_f32, C);
                    }

                    // Scalar tail for remaining columns
                    for (; jr < NR; ++jr)
                    {
                        process_n_columns<1>(jr, ir, kb, ldc, sA, dA, a_vec, b_vec, b_scales_f32, C);
                    }
                }
            } // End K-loop
        } // End microkernel_full_sa

        /**
         * @brief Dispatcher: Full MR×NR microkernel (selects implementation based on debugEnv)
         *
         * @param K_blocks Number of K blocks to process
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        static void microkernel_full(
            int K_blocks,
            int i_base,
            int kc_start,
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            // Print compensation mode once at first call (for verification)
            static bool mode_logged = false;
            if (!mode_logged)
            {
                if (debugEnv().gemm.use_sa_compensation)
                {
                    std::cerr << "[Q8_1 GEMM] Using sA-based compensation (optimized, no quantization)" << std::endl;
                }
                else
                {
                    std::cerr << "[Q8_1 GEMM] Using sum_qs compensation (baseline, with quantization)" << std::endl;
                }
                mode_logged = true;
            }

            if (debugEnv().gemm.use_sa_compensation)
            {
                microkernel_full_sa(K_blocks, i_base, kc_start, A_decodable, B_packed, C, ldc);
            }
            else
            {
                microkernel_full_sumqs(K_blocks, i_base, kc_start, A_decodable, B_packed, C, ldc);
            }
        }

        /**
         * @brief Edge-case microkernel (partial tiles)
         *
         * Scalar fallback for non-multiple-of-8 dimensions
         */
        /**
         * @brief Edge microkernel: Handles cases where M < MR or N < NR
         *
         * @param m_block Actual number of rows to process (≤ MR)
         * @param n_block Actual number of columns to process (≤ NR)
         * @param K_blocks Number of K blocks (32-element blocks)
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        static void microkernel_edge(
            int m_block, int n_block, int K_blocks, int i_base, int kc_start,
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();
            const int B_K_blocks = B_packed.K_blocks; // Use packed panel's K dimension

            // Pre-allocate storage for vectorized K-block reduction
            // (amortize allocation cost across all m_block×n_block elements)
            std::vector<int32_t> block_dots_storage(K_blocks);
            std::vector<int32_t> sum_qs_storage(K_blocks); // Q8_1 compensation: sum_qs per block
            std::vector<uint16_t> a_scales_storage(K_blocks);
            std::vector<uint16_t> b_scales_storage(K_blocks);

            for (int i = 0; i < m_block; ++i)
            {
                for (int j = 0; j < n_block; ++j)
                {
                    // PHASE 1: Compute all K-block dot products using dpbusd (VECTORIZED!)
                    // Determine which jr_pair contains column j (needed for B access)
                    int jr_pair = j / 2;
                    int col_offset = (j % 2) * 32; // 0 for even j, 32 for odd j

                    for (int kb = 0; kb < K_blocks; ++kb)
                    {
                        // Decode A block via IQ8_1Decodable interface
                        const Q8_1Block *a_block_ptr = A_decodable->decode_to_q8_1(i_base + i, kc_start + kb);
                        const Q8_1Block &a_block = *a_block_ptr;

                        // Store scales for vectorized reduction
                        a_scales_storage[kb] = a_block.d;
                        b_scales_storage[kb] = B_scales[j * B_K_blocks + kb];

                        // Q8_1 COMPENSATION: Extract sum_qs from Q8_1 block
                        // sum_qs = sum_a / a_scale (where sum_a and a_scale are FP16)
                        float sum_a_f32 = fp16_to_fp32(a_block.s);
                        float a_scale_f32 = fp16_to_fp32(a_block.d);
                        float sum_qs_f32 = sum_a_f32 / std::max(a_scale_f32, 1e-10f);
                        sum_qs_storage[kb] = static_cast<int32_t>(std::round(sum_qs_f32));

                        // OPTIMIZATION: Use dpbusd for 32-element dot product (32 scalar ops → 1 SIMD instruction!)
                        // Calculate base address for this jr_pair's 64-byte ZMM
                        const uint8_t *zmm_base = B_quants + (kb * (NR / 2) + jr_pair) * 64;

                        // Load A vector (32×INT8) and zero-extend to 512 bits
                        __m256i a_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block.qs));
                        __m512i a_vec = _mm512_castsi256_si512(a_ymm);

                        // Load B vector (32×UINT8) and zero-extend to 512 bits
                        __m256i b_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&zmm_base[col_offset]));
                        __m512i b_vec = _mm512_castsi256_si512(b_ymm);

                        // Vectorized dot product using dpbusd
                        __m512i acc = _mm512_setzero_si512();
                        acc = _mm512_dpbusd_epi32(acc, b_vec, a_vec);

                        // Horizontal reduction to scalar
                        block_dots_storage[kb] = _mm512_reduce_add_epi32(acc);
                    }

                    // PHASE 2: Vectorized K-block reduction (16-wide, 8-wide, 4-wide, scalar tail)
                    // Q8_1 Formula: result = Σ_kb ((block_dot[kb] - 128*sum_qs[kb]) * a_scale[kb] * b_scale[kb])
                    // NOTE: dpbusd computes Σ((B+128)*A), so we must subtract 128*Σ(A) to get Σ(B*A)
                    float result = 0.0f;
                    int kb = 0;

                    // 16-wide vectorization (AVX-512)
                    for (; kb + 16 <= K_blocks; kb += 16)
                    {
                        // Load 16 block_dots (INT32) and convert to FP32
                        __m512i dots_i32 = _mm512_loadu_si512(&block_dots_storage[kb]);
                        __m512 dots_f32 = _mm512_cvtepi32_ps(dots_i32);

                        // Load 16 sum_qs (INT32) and convert to FP32
                        __m512i sum_qs_i32 = _mm512_loadu_si512(&sum_qs_storage[kb]);
                        __m512 sum_qs_f32 = _mm512_cvtepi32_ps(sum_qs_i32);

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        const __m512 compensation_const = _mm512_set1_ps(128.0f);
                        __m512 compensated = _mm512_fnmadd_ps(sum_qs_f32, compensation_const, dots_f32);

                        // Load 16 A scales (FP16) and convert to FP32
                        __m256i a_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a_scales_storage[kb]));
                        __m512 a_scales_f32 = _mm512_cvtph_ps(a_scales_fp16);

                        // Load 16 B scales (FP16) and convert to FP32
                        __m256i b_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b_scales_storage[kb]));
                        __m512 b_scales_f32 = _mm512_cvtph_ps(b_scales_fp16);

                        // Compute: compensated * a_scale * b_scale
                        __m512 scaled = _mm512_mul_ps(compensated, a_scales_f32);
                        scaled = _mm512_mul_ps(scaled, b_scales_f32);

                        // Accumulate (horizontal reduction)
                        result += _mm512_reduce_add_ps(scaled);
                    }

                    // 8-wide vectorization (AVX2)
                    for (; kb + 8 <= K_blocks; kb += 8)
                    {
                        // Load 8 block_dots (INT32) and convert to FP32
                        __m256i dots_i32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&block_dots_storage[kb]));
                        __m256 dots_f32 = _mm256_cvtepi32_ps(dots_i32);

                        // Load 8 sum_qs (INT32) and convert to FP32
                        __m256i sum_qs_i32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&sum_qs_storage[kb]));
                        __m256 sum_qs_f32 = _mm256_cvtepi32_ps(sum_qs_i32);

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        const __m256 compensation_const = _mm256_set1_ps(128.0f);
                        __m256 compensated = _mm256_fnmadd_ps(sum_qs_f32, compensation_const, dots_f32);

                        // Load 8 A scales (FP16) and convert to FP32
                        __m128i a_scales_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&a_scales_storage[kb]));
                        __m256 a_scales_f32 = _mm256_cvtph_ps(a_scales_fp16);

                        // Load 8 B scales (FP16) and convert to FP32
                        __m128i b_scales_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&b_scales_storage[kb]));
                        __m256 b_scales_f32 = _mm256_cvtph_ps(b_scales_fp16);

                        // Compute: compensated * a_scale * b_scale
                        __m256 scaled = _mm256_mul_ps(compensated, a_scales_f32);
                        scaled = _mm256_mul_ps(scaled, b_scales_f32);

                        // Horizontal reduction (AVX2-compatible)
                        __m128 lo = _mm256_castps256_ps128(scaled);
                        __m128 hi = _mm256_extractf128_ps(scaled, 1);
                        __m128 sum4 = _mm_add_ps(lo, hi);
                        __m128 shuf = _mm_movehdup_ps(sum4);
                        __m128 sum2 = _mm_add_ps(sum4, shuf);
                        shuf = _mm_movehl_ps(shuf, sum2);
                        __m128 sum1 = _mm_add_ss(sum2, shuf);
                        result += _mm_cvtss_f32(sum1);
                    }

                    // 4-wide vectorization (AVX2)
                    for (; kb + 4 <= K_blocks; kb += 4)
                    {
                        // Load 4 block_dots (INT32) and convert to FP32
                        __m128i dots_i32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&block_dots_storage[kb]));
                        __m128 dots_f32 = _mm_cvtepi32_ps(dots_i32);

                        // Load 4 sum_qs (INT32) and convert to FP32
                        __m128i sum_qs_i32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&sum_qs_storage[kb]));
                        __m128 sum_qs_f32 = _mm_cvtepi32_ps(sum_qs_i32);

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        const __m128 compensation_const = _mm_set1_ps(128.0f);
                        __m128 compensated = _mm_fnmadd_ps(sum_qs_f32, compensation_const, dots_f32);

                        // Load 4 A scales (FP16) and convert to FP32
                        __m128i a_scales_fp16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&a_scales_storage[kb]));
                        __m128 a_scales_f32 = _mm_cvtph_ps(a_scales_fp16);

                        // Load 4 B scales (FP16) and convert to FP32
                        __m128i b_scales_fp16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&b_scales_storage[kb]));
                        __m128 b_scales_f32 = _mm_cvtph_ps(b_scales_fp16);

                        // Compute: compensated * a_scale * b_scale
                        __m128 scaled = _mm_mul_ps(compensated, a_scales_f32);
                        scaled = _mm_mul_ps(scaled, b_scales_f32);

                        // Horizontal reduction: 4→1
                        __m128 shuf = _mm_movehdup_ps(scaled);
                        __m128 sum2 = _mm_add_ps(scaled, shuf);
                        shuf = _mm_movehl_ps(shuf, sum2);
                        __m128 sum1 = _mm_add_ss(sum2, shuf);
                        result += _mm_cvtss_f32(sum1);
                    }

                    // Scalar tail (0-3 blocks remaining)
                    for (; kb < K_blocks; ++kb)
                    {
                        int32_t block_dot = block_dots_storage[kb];
                        int32_t sum_qs_val = sum_qs_storage[kb];

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        float compensated = static_cast<float>(block_dot - sum_qs_val * 128);

                        float a_scale = fp16_to_fp32(a_scales_storage[kb]);
                        float b_scale = fp16_to_fp32(b_scales_storage[kb]);
                        result += compensated * a_scale * b_scale;
                    }

                    C[i * ldc + j] = result; // FIXED: Use assignment (=) not accumulation (+=)
                }
            }
        }
    };

    // Type aliases for common microkernel sizes
    // 32×128 with prefetch=4 is optimal (+5.3% over 32×64, determined by parameter sweep Nov 2025)
    // Default parameters optimized via comprehensive benchmark sweep (64 configurations tested)
    // Q8_1 GEMM uses same microkernel sizes but with IQ8_1Decodable activations
    // VECTOR_WIDTH is hardcoded to 16 (matches __m512 register capacity)

    // Default configuration: 2-column unrolling (balanced performance across all M)
    using Q8_1GemmKernel = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 2>; // Default: JR_UNROLL=2 (optimal: 496 GFLOPS @ M=8192)

    // Alternative unrolling configurations (for autotuner to explore)
    using Q8_1GemmKernel_1col = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 1>; // JR_UNROLL=1: Minimal ILP (baseline)
    using Q8_1GemmKernel_4col = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 4>; // JR_UNROLL=4: +5.5% @ M=4096, -20% @ M=512
    using Q8_1GemmKernel_6col = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 6>; // JR_UNROLL=6: High ILP (untested)
    using Q8_1GemmKernel_8col = Q8_1GemmKernelTemplate<32, 128, 4, 0, 0, 8>; // JR_UNROLL=8: Maximum ILP (may hurt due to register spilling)

    // Legacy microkernel size variants
    using Q8_1GemmKernel_32x64 = Q8_1GemmKernelTemplate<32, 64, 4, 0, 0, 2>; // Previous default (439.2 GFLOPS)
    using Q8_1GemmKernel_8x8 = Q8_1GemmKernelTemplate<8, 8, 2, 0, 0, 2>;     // Small baseline
    using Q8_1GemmKernel_16x16 = Q8_1GemmKernelTemplate<16, 16, 2, 0, 0, 2>; // Medium microkernel
    using Q8_1GemmKernel_32x32 = Q8_1GemmKernelTemplate<32, 32, 2, 0, 0, 2>; // Square microkernel

} // namespace llaminar2
