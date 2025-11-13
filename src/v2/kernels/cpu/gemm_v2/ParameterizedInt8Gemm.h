/**
 * @file ParameterizedInt8Gemm.h
 * @brief Template-based parameterized INT8 GEMM microkernel (OneDNN-inspired)
 *
 * Mirrors OneDNN's hierarchical unrolling strategy:
 * - Primary kernel: 48×8 (like OneDNN's IGEMM_UNROLL_M_ × IGEMM_UNROLL_N_)
 * - Smaller kernels: 24×8, 12×8, 6×8, 3×8 for remainder handling
 * - Template parameters allow compile-time specialization
 *
 * Key OneDNN insights:
 * - Larger M unrolling (48 vs our 6) amortizes fixed costs better
 * - Smaller N unrolling (8 vs our 16) fits more M rows in registers
 * - Hierarchical fallback for edge cases
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <memory>

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {
            namespace cpu
            {

                /**
                 * @class ParameterizedInt8GemmKernel
                 * @brief Template-based INT8 GEMM microkernel with compile-time MR/NR configuration
                 *
                 * @tparam MR Number of rows in microkernel (must be multiple of 16 for ZMM packing)
                 * @tparam NR Number of columns in microkernel
                 * @tparam PREFETCH_A_DIST Prefetch distance for A matrix in bytes (OneDNN: 160)
                 * @tparam PREFETCH_B_DIST Prefetch distance for B matrix in bytes (OneDNN: 128)
                 * @tparam PREFETCH_C_DIST Prefetch distance for C matrix in bytes (OneDNN: 64)
                 *
                 * OneDNN uses:
                 * - Primary: MR=48, NR=8 (3 ZMM registers for A, 8 output registers per A row)
                 * - This gives 48×8 = 384 INT32 outputs across 3×8 = 24 ZMM registers
                 * - Prefetching: 160 bytes ahead for A (5 cache lines), 128 bytes for B (4 cache lines)
                 */
                template <int MR, int NR, int PREFETCH_A_DIST = 160, int PREFETCH_B_DIST = 128, int PREFETCH_C_DIST = 64>
                class ParameterizedInt8GemmKernel
                {
                public:
                    static_assert(MR % 16 == 0, "MR must be multiple of 16 (ZMM holds 16 INT32s)");
                    static_assert(NR > 0 && NR <= 16, "NR must be in range [1, 16]");
                    static_assert((MR / 16) * NR <= 30, "Total register count exceeds available ZMMs");

                    static constexpr int M_VECS = MR / 16; // Number of ZMM vectors for M dimension

                    // Prefetch hint intrinsics (OneDNN pattern)
                    static inline void prefetch_a(const void *addr)
                    {
                        _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0); // L1 cache
                    }

                    static inline void prefetch_b(const void *addr)
                    {
                        _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0); // L1 cache
                    }

                    static inline void prefetch_c(const void *addr)
                    {
                        _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_ET1); // Write prefetch (L2, exclusive)
                    }

                    /**
                     * @brief Pack B panel for VNNI: layout [K/4][NR][4] (NR*4 bytes per K-block).
                     *
                     * OneDNN approach: Convert signed→unsigned during packing (XOR with 0x80).
                     * This enables use of vpdpbusd (unsigned×signed) for signed×signed GEMM.
                     *
                     * Phase 4: Vectorized implementation using AVX-512
                     * - Process 16 K-elements (4 K-steps) at a time
                     * - SIMD s8→u8 conversion (64 elements in parallel)
                     * - SIMD transpose using byte shuffles
                     * - Expected: +15% improvement from eliminating scalar bottleneck
                     */
                    static void pack_B_panel(const int8_t *__restrict__ B_src,
                                             int8_t *__restrict__ B_packed,
                                             int K,
                                             int ldb,
                                             bool convert_s8_to_u8 = true)
                    {
                        // XOR constant for s8→u8 conversion (broadcast to all lanes)
                        const __m512i xor_mask = _mm512_set1_epi8(static_cast<int8_t>(0x80));

                        // Process K in blocks of 16 (4 K-steps) for better vectorization
                        int k = 0;
                        for (; k + 15 < K; k += 16)
                        {
                            // Process 4 K-steps (16 elements) at once
                            // Layout: [K-step0][K-step1][K-step2][K-step3], each is NR×4 bytes

                            // Load NR rows × 16 columns (using 128-bit loads for each row)
                            alignas(64) int8_t temp[NR][16];
                            for (int j = 0; j < NR; ++j)
                            {
                                // Load 16 bytes from row j
                                __m128i row = _mm_loadu_si128(reinterpret_cast<const __m128i *>(B_src + j * ldb + k));
                                _mm_store_si128(reinterpret_cast<__m128i *>(temp[j]), row);
                            }

                            // Transpose and convert s8→u8 in parallel
                            // Target layout: [k0:k3][col0-7], [k4:k7][col0-7], [k8:k11][col0-7], [k12:k15][col0-7]

                            if (convert_s8_to_u8)
                            {
                                // Process each K-step (4 elements per column)
                                for (int ks = 0; ks < 4; ks++)
                                {
                                    // Build vector: [col0[k:k+3], col1[k:k+3], ..., colNR-1[k:k+3]]
                                    alignas(64) int8_t temp_step[64]; // Support up to NR=16
                                    for (int j = 0; j < NR; ++j)
                                    {
                                        for (int kk = 0; kk < 4; ++kk)
                                        {
                                            temp_step[j * 4 + kk] = temp[j][ks * 4 + kk];
                                        }
                                    }

                                    // Convert s8→u8 using SIMD XOR
                                    if (NR <= 8)
                                    {
                                        __m256i data = _mm256_load_si256(reinterpret_cast<const __m256i *>(temp_step));
                                        data = _mm256_xor_si256(data, _mm512_castsi512_si256(xor_mask));

                                        // Store to packed buffer
                                        int block_index = ((k + ks * 4) / 4) * NR * 4;
                                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(B_packed + block_index), data);
                                    }
                                    else
                                    {
                                        // NR > 8: use full ZMM register
                                        __m512i data = _mm512_load_si512(reinterpret_cast<const __m512i *>(temp_step));
                                        data = _mm512_xor_si512(data, xor_mask);

                                        int block_index = ((k + ks * 4) / 4) * NR * 4;
                                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(B_packed + block_index), data);
                                    }
                                }
                            }
                            else
                            {
                                // No conversion, just transpose
                                for (int ks = 0; ks < 4; ks++)
                                {
                                    alignas(64) int8_t temp_step[64]; // Support up to NR=16
                                    for (int j = 0; j < NR; ++j)
                                    {
                                        for (int kk = 0; kk < 4; ++kk)
                                        {
                                            temp_step[j * 4 + kk] = temp[j][ks * 4 + kk];
                                        }
                                    }

                                    int block_index = ((k + ks * 4) / 4) * NR * 4;
                                    if (NR <= 8)
                                    {
                                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(B_packed + block_index),
                                                            _mm256_load_si256(reinterpret_cast<const __m256i *>(temp_step)));
                                    }
                                    else
                                    {
                                        _mm512_storeu_si512(reinterpret_cast<__m512i *>(B_packed + block_index),
                                                            _mm512_load_si512(reinterpret_cast<const __m512i *>(temp_step)));
                                    }
                                }
                            }
                        }

                        // Handle remainder K elements (scalar fallback)
                        for (; k < K; k += 4)
                        {
                            int block_index = (k / 4) * NR * 4;
                            int8_t *dst = B_packed + block_index;

                            for (int j = 0; j < NR; ++j)
                            {
                                for (int kk = 0; kk < 4; ++kk)
                                {
                                    int8_t val = B_src[j * ldb + k + kk];
                                    if (convert_s8_to_u8)
                                    {
                                        val ^= static_cast<int8_t>(0x80);
                                    }
                                    dst[j * 4 + kk] = val;
                                }
                            }
                        }
                    }

                    /**
                     * @brief Microkernel: MR×NR output tile with 4× K-loop unrolling
                     *
                     * OneDNN pattern:
                     * - B converted to unsigned during packing (XOR 0x80)
                     * - Compensation: result -= sum(A_row) × 128
                     *
                     * Phase 3 optimization: Compensation is precomputed outside this function
                     * and passed in, eliminating hot-loop overhead.
                     *
                     * @param A_compensation Precomputed compensation for each row (sum(A_row) × 128)
                     */
                    static void microkernel(const int8_t *__restrict__ A,
                                            const int8_t *__restrict__ Bpack,
                                            int32_t *__restrict__ C,
                                            int K,
                                            int lda,
                                            int ldc,
                                            const int32_t *__restrict__ A_compensation = nullptr)
                    {
                        // Accumulators for MR×NR output tile
                        __m512i c_regs[M_VECS][NR];
                        for (int i = 0; i < M_VECS; ++i)
                        {
                            for (int j = 0; j < NR; ++j)
                            {
                                c_regs[i][j] = _mm512_setzero_si512();
                            }
                        }

                        // Phase 3: Load precomputed compensation (if provided)
                        // Otherwise compute inline (backward compatibility)
                        __m512i comp_vec[M_VECS];
                        bool use_precomputed = (A_compensation != nullptr);

                        if (use_precomputed)
                        {
                            // Load precomputed compensation values
                            for (int i = 0; i < M_VECS; ++i)
                            {
                                int row_base = i * 16;
                                alignas(64) int32_t comp_data[16];
                                for (int r = 0; r < 16; ++r)
                                {
                                    if (row_base + r < MR)
                                    {
                                        comp_data[r] = A_compensation[row_base + r];
                                    }
                                    else
                                    {
                                        comp_data[r] = 0;
                                    }
                                }
                                comp_vec[i] = _mm512_load_si512(reinterpret_cast<const __m512i *>(comp_data));
                            }
                        }

                        // Accumulators for sum(A) per row (only needed if not precomputed)
                        __m512i sum_a[M_VECS];
                        if (!use_precomputed)
                        {
                            for (int i = 0; i < M_VECS; ++i)
                            {
                                sum_a[i] = _mm512_setzero_si512();
                            }
                        }

                        const __m512i ones_u8 = _mm512_set1_epi8(1);

                        // Main loop: 4× unrolled (16 K elements per iteration)
                        int k = 0;
                        for (; k + 15 < K; k += 16)
                        {
                            // Prefetch A for next iteration (OneDNN pattern: 5 cache lines ahead = 160 bytes)
                            // Each row of A is lda bytes, prefetch PREFETCH_A_DIST bytes ahead in current row
                            if (PREFETCH_A_DIST > 0 && k + PREFETCH_A_DIST < lda)
                            {
                                for (int i = 0; i < M_VECS && i * 16 < MR; ++i)
                                {
                                    prefetch_a(A + i * 16 * lda + k + PREFETCH_A_DIST);
                                }
                            }

                            // Prefetch B for next iteration (OneDNN pattern: 4 cache lines ahead = 128 bytes)
                            // Bpack layout: [K/4][NR][4], each K-block is NR*4 bytes
                            if (PREFETCH_B_DIST > 0)
                            {
                                int k_blocks_ahead = PREFETCH_B_DIST / (NR * 4); // How many K-blocks ahead
                                int k_prefetch = k + k_blocks_ahead * 4;         // K index to prefetch
                                if (k_prefetch < K)
                                {
                                    prefetch_b(Bpack + (k_prefetch / 4) * NR * 4);
                                }
                            }

                            // Load all 4 B blocks (each is NR*4 bytes)
                            __m512i b[4];
                            for (int h = 0; h < 4; ++h)
                            {
                                // For NR < 16, we need to handle partial loads
                                if constexpr (NR == 16)
                                {
                                    b[h] = _mm512_loadu_si512(
                                        reinterpret_cast<const __m512i *>(Bpack + (k / 4 + h) * 64));
                                }
                                else if constexpr (NR == 8)
                                {
                                    // Load 32 bytes (8 columns × 4 bytes)
                                    __m256i b_half = _mm256_loadu_si256(
                                        reinterpret_cast<const __m256i *>(Bpack + (k / 4 + h) * 32));
                                    b[h] = _mm512_castsi256_si512(b_half);
                                }
                                else
                                {
                                    // For other sizes, use masked load or scalar fallback
                                    // (Simplified: broadcast from memory)
                                    b[h] = _mm512_loadu_si512(
                                        reinterpret_cast<const __m512i *>(Bpack + (k / 4 + h) * NR * 4));
                                }
                            }

                            // Process all 4 K-blocks
                            for (int h = 0; h < 4; ++h)
                            {
                                // Load A for all MR rows
                                __m512i a_vec[M_VECS];
                                for (int i = 0; i < M_VECS; ++i)
                                {
                                    // Interleaved A prefetch (OneDNN pattern: prefetch next row's data)
                                    if (PREFETCH_A_DIST > 0 && i == 0 && h < 3)
                                    {
                                        int row_prefetch = (i + 1) * 16;  // Next M_VEC block
                                        int k_prefetch = k + (h + 1) * 4; // Next h iteration's K
                                        if (row_prefetch < MR && k_prefetch + PREFETCH_A_DIST < lda)
                                        {
                                            prefetch_a(A + row_prefetch * lda + k_prefetch + PREFETCH_A_DIST);
                                        }
                                    }

                                    // Load 16 rows worth of A (each row contributes 4 bytes = 1 int32)
                                    // Use AVX-512 gather for vectorized strided loads instead of 16 scalar loads
                                    int row_base = i * 16;

                                    // Build index vector for gather: [0*lda, 1*lda, 2*lda, ..., 15*lda]
                                    __m512i gather_indices = _mm512_setr_epi32(
                                        0 * lda, 1 * lda, 2 * lda, 3 * lda, 4 * lda, 5 * lda, 6 * lda, 7 * lda,
                                        8 * lda, 9 * lda, 10 * lda, 11 * lda, 12 * lda, 13 * lda, 14 * lda, 15 * lda);

                                    // Gather 16 INT32 values (4 INT8 bytes each) from strided rows
                                    // Base address: A + row_base*lda + k + h*4
                                    const int32_t *gather_base = reinterpret_cast<const int32_t *>(
                                        A + row_base * lda + k + h * 4);

                                    // Create mask for rows that are in bounds
                                    uint16_t mask = (row_base + 16 <= MR) ? 0xFFFF : ((1u << (MR - row_base)) - 1);

                                    // Gather with scale=1 (addresses are already int32* spaced)
                                    a_vec[i] = _mm512_mask_i32gather_epi32(
                                        _mm512_setzero_si512(), // src (zero for out-of-bounds)
                                        (__mmask16)mask,        // mask
                                        gather_indices,         // indices (in bytes)
                                        gather_base,            // base address
                                        1                       // scale (1 = no scaling, indices are byte offsets)
                                    );
                                }

                                // Compute MR×NR outputs via dpbusd
                                for (int j = 0; j < NR; ++j)
                                {
                                    // Broadcast B[j] to all lanes (store then broadcast - avoids constexpr issues)
                                    alignas(64) int32_t b_data[16];
                                    if constexpr (NR == 16)
                                    {
                                        _mm512_store_si512(reinterpret_cast<__m512i *>(b_data), b[h]);
                                    }
                                    else if constexpr (NR == 8)
                                    {
                                        _mm256_store_si256(reinterpret_cast<__m256i *>(b_data),
                                                           _mm512_castsi512_si256(b[h]));
                                    }
                                    else
                                    {
                                        _mm512_store_si512(reinterpret_cast<__m512i *>(b_data), b[h]);
                                    }
                                    __m512i b_broadcast = _mm512_set1_epi32(b_data[j]);

                                    for (int i = 0; i < M_VECS; ++i)
                                    {
                                        c_regs[i][j] = _mm512_dpbusd_epi32(c_regs[i][j], b_broadcast, a_vec[i]);
                                    }
                                }

                                // Accumulate sum(A) for compensation (only if not precomputed)
                                if (!use_precomputed)
                                {
                                    for (int i = 0; i < M_VECS; ++i)
                                    {
                                        sum_a[i] = _mm512_dpbusd_epi32(sum_a[i], ones_u8, a_vec[i]);
                                    }
                                }
                            }
                        }

                        // Remainder loop: process remaining K-blocks one at a time
                        for (; k < K; k += 4)
                        {
                            __m512i b;
                            if constexpr (NR == 16)
                            {
                                b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4) * 64));
                            }
                            else if constexpr (NR == 8)
                            {
                                __m256i b_half = _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i *>(Bpack + (k / 4) * 32));
                                b = _mm512_castsi256_si512(b_half);
                            }
                            else
                            {
                                b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bpack + (k / 4) * NR * 4));
                            }

                            // Load A and compute
                            __m512i a_vec[M_VECS];
                            for (int i = 0; i < M_VECS; ++i)
                            {
                                int row_base = i * 16;

                                // Build index vector for gather: [0*lda, 1*lda, ..., 15*lda]
                                __m512i gather_indices = _mm512_setr_epi32(
                                    0 * lda, 1 * lda, 2 * lda, 3 * lda, 4 * lda, 5 * lda, 6 * lda, 7 * lda,
                                    8 * lda, 9 * lda, 10 * lda, 11 * lda, 12 * lda, 13 * lda, 14 * lda, 15 * lda);

                                const int32_t *gather_base = reinterpret_cast<const int32_t *>(
                                    A + row_base * lda + k);

                                uint16_t mask = (row_base + 16 <= MR) ? 0xFFFF : ((1u << (MR - row_base)) - 1);

                                a_vec[i] = _mm512_mask_i32gather_epi32(
                                    _mm512_setzero_si512(),
                                    (__mmask16)mask,
                                    gather_indices,
                                    gather_base,
                                    1);
                            }

                            for (int j = 0; j < NR; ++j)
                            {
                                // Store then broadcast to avoid constexpr issues
                                alignas(64) int32_t b_data[16];
                                if constexpr (NR == 16)
                                {
                                    _mm512_store_si512(reinterpret_cast<__m512i *>(b_data), b);
                                }
                                else if constexpr (NR == 8)
                                {
                                    _mm256_store_si256(reinterpret_cast<__m256i *>(b_data),
                                                       _mm512_castsi512_si256(b));
                                }
                                else
                                {
                                    _mm512_store_si512(reinterpret_cast<__m512i *>(b_data), b);
                                }
                                __m512i b_broadcast = _mm512_set1_epi32(b_data[j]);

                                for (int i = 0; i < M_VECS; ++i)
                                {
                                    c_regs[i][j] = _mm512_dpbusd_epi32(c_regs[i][j], b_broadcast, a_vec[i]);
                                }
                            }

                            // Accumulate sum(A) for compensation (only if not precomputed)
                            if (!use_precomputed)
                            {
                                for (int i = 0; i < M_VECS; ++i)
                                {
                                    sum_a[i] = _mm512_dpbusd_epi32(sum_a[i], ones_u8, a_vec[i]);
                                }
                            }
                        }

                        // Apply compensation: result -= sum(A_row) × 128
                        // Phase 3: Use precomputed values if available, otherwise compute inline
                        if (!use_precomputed)
                        {
                            // Compute compensation inline (backward compatibility)
                            for (int i = 0; i < M_VECS; ++i)
                            {
                                // Extract the 16 sum values (one per row)
                                alignas(64) int32_t row_sums[16];
                                _mm512_store_si512(reinterpret_cast<__m512i *>(row_sums), sum_a[i]);

                                // Multiply by 128 to get compensation
                                for (int r = 0; r < 16; ++r)
                                {
                                    row_sums[r] *= 128;
                                }

                                // Load compensation vector and store for later use
                                comp_vec[i] = _mm512_load_si512(reinterpret_cast<const __m512i *>(row_sums));
                            }
                        }

                        // Subtract compensation from all outputs (precomputed or computed inline)
                        for (int i = 0; i < M_VECS; ++i)
                        {
                            for (int j = 0; j < NR; ++j)
                            {
                                c_regs[i][j] = _mm512_sub_epi32(c_regs[i][j], comp_vec[i]);
                            }
                        }

                        // Store results to C
                        for (int i = 0; i < M_VECS; ++i)
                        {
                            for (int j = 0; j < NR; ++j)
                            {
                                int row_base = i * 16;

                                // Prefetch C for write (OneDNN pattern: write-intent prefetch)
                                if (i == 0 && j == 0 && row_base + PREFETCH_C_DIST / (ldc * sizeof(int32_t)) < MR)
                                {
                                    prefetch_c(C + (row_base + PREFETCH_C_DIST / (ldc * sizeof(int32_t))) * ldc);
                                }

                                alignas(64) int32_t c_data[16];
                                _mm512_store_si512(reinterpret_cast<__m512i *>(c_data), c_regs[i][j]);

                                for (int r = 0; r < 16 && row_base + r < MR; ++r)
                                {
                                    C[(row_base + r) * ldc + j] = c_data[r];
                                }
                            }
                        }
                    }

                    /**
                     * @brief Top-level GEMM driver with hierarchical tiling
                     *
                     * OneDNN pattern:
                     * - Outer loop: N dimension (NR-sized panels)
                     * - Inner loop: M dimension (MR-sized tiles with remainder handling)
                     *
                     * OpenMP Parallelization:
                     * - Parallelize over M dimension (rows of A)
                     * - Static scheduling for load balance
                     * - Each thread processes MR rows independently
                     */
                    static void gemm(int M, int N, int K,
                                     const int8_t *__restrict__ A, int lda,
                                     const int8_t *__restrict__ B, int ldb,
                                     int32_t *__restrict__ C, int ldc)
                    {
                        // Phase 3: Precompute A compensation: sum(A_row) × 128
                        // This eliminates the need to compute it in the microkernel hot loop
                        std::vector<int32_t> A_compensation(M);

#pragma omp parallel for schedule(static)
                        for (int i = 0; i < M; ++i)
                        {
                            int32_t sum = 0;
                            for (int k = 0; k < K; ++k)
                            {
                                sum += static_cast<int32_t>(A[i * lda + k]);
                            }
                            A_compensation[i] = sum * 128; // Multiply by zero-point offset
                        }

                        // Process N in NR-column panels
                        for (int j = 0; j < N; j += NR)
                        {
                            int n_block = std::min(N - j, NR);

                            // Pack B panel if full width and K aligned
                            std::unique_ptr<int8_t[]> Bpack;
                            if (n_block == NR && (K % 4 == 0))
                            {
                                Bpack.reset(new int8_t[(size_t)K * NR]);
                                pack_B_panel(B + j * ldb, Bpack.get(), K, ldb, true);

// Process M in MR-row tiles (PARALLELIZED)
#pragma omp parallel for schedule(static)
                                for (int i = 0; i < M; i += MR)
                                {
                                    int m_block = std::min(M - i, MR);

                                    if (m_block == MR)
                                    {
                                        microkernel(A + i * lda, Bpack.get(), C + i * ldc + j,
                                                    K, lda, ldc, A_compensation.data() + i);
                                    }
                                    else
                                    {
                                        // Scalar fallback for partial M tiles
                                        for (int ii = 0; ii < m_block; ++ii)
                                        {
                                            for (int jj = 0; jj < n_block; ++jj)
                                            {
                                                int32_t sum = 0;
                                                for (int k = 0; k < K; ++k)
                                                {
                                                    sum += static_cast<int32_t>(A[(i + ii) * lda + k]) *
                                                           static_cast<int32_t>(B[(j + jj) * ldb + k]);
                                                }
                                                C[(i + ii) * ldc + (j + jj)] = sum;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                // Scalar fallback for partial N tiles or K misalignment
                                for (int i = 0; i < M; ++i)
                                {
                                    for (int jj = 0; jj < n_block; ++jj)
                                    {
                                        int32_t sum = 0;
                                        for (int k = 0; k < K; ++k)
                                        {
                                            sum += static_cast<int32_t>(A[i * lda + k]) *
                                                   static_cast<int32_t>(B[(j + jj) * ldb + k]);
                                        }
                                        C[i * ldc + (j + jj)] = sum;
                                    }
                                }
                            }
                        }
                    }
                };

                // Instantiate common configurations (like OneDNN)
                // Default prefetch distances: A=160, B=128, C=64 (OneDNN values)
                using Int8Gemm_48x8 = ParameterizedInt8GemmKernel<48, 8, 160, 128, 64>;  // OneDNN primary
                using Int8Gemm_32x8 = ParameterizedInt8GemmKernel<32, 8, 160, 128, 64>;  // Large
                using Int8Gemm_16x8 = ParameterizedInt8GemmKernel<16, 8, 160, 128, 64>;  // Medium
                using Int8Gemm_6x16 = ParameterizedInt8GemmKernel<16, 16, 160, 128, 64>; // Original (but 16-aligned)

                // Variants with different prefetch distances for tuning
                using Int8Gemm_16x16_Prefetch256 = ParameterizedInt8GemmKernel<16, 16, 256, 192, 96>; // Aggressive
                using Int8Gemm_16x16_Prefetch64 = ParameterizedInt8GemmKernel<16, 16, 64, 64, 32>;    // Conservative
                using Int8Gemm_16x16_NoPrefetch = ParameterizedInt8GemmKernel<16, 16, 0, 0, 0>;       // Baseline (no prefetch)

            } // namespace cpu
        } // namespace kernels
    } // namespace v2
} // namespace llaminar
