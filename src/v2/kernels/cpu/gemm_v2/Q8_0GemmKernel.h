/**
 * @file Q8_0GemmKernel.h
 * @brief Q8_0 × Q8_0 → FP32 GEMM kernel using standard Q8_0Tensor interface
 * @author David Sanftenberg
 *
 * Computes C_fp32 = A_q8 × B_q8 where both A and B are Q8_0 quantized tensors.
 * Uses inline compensation following Phase 3 learnings (precomputation inefficient).
 *
 * Q8_0 Format:
 * - Blocks of 32 int8 values
 * - Per-block FP16 scale (uint16_t d)
 * - Block size: 34 bytes (2 + 32)
 *
 * Formula: C = (A_scales ⊗ B_scales) ⊙ (A_quants × B_quants - compensation)
 * Where compensation = sum(A_row) × 128 (computed inline via dpbusd)
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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar2
{

    // Forward declarations (in case includes don't provide them)
    class Q8_0Tensor;

    /**
     * @class Q8_0GemmKernel
     * @brief High-performance Q8_0 × Q8_0 GEMM using parameterized MR×NR AVX-512 microkernel
     *
     * Key Design Decisions:
     * 1. Parameterized microkernel (MR×NR template) - tunable for different architectures
     * 2. Inline compensation - computed during GEMM (free via ILP)
     * 3. Scale packing - B scales packed alongside quantized values
     * 4. No A-packing - Q8_0 blocks already cache-friendly
     * 5. B packing - transpose + s8→u8 for VNNI layout + scale extraction
     * 6. NC/KC blocking - cache-friendly blocking for large matrices
     *
     * @tparam MR_PARAM M register blocking (rows per microkernel)
     * @tparam NR_PARAM N register blocking (columns per microkernel)
     * @tparam PREFETCH_A_PARAM A block prefetch distance (0-5, default 4)
     * @tparam VECTOR_WIDTH_PARAM K-blocks processed per post-processing iteration (8 optimal, 16 safe for all sizes)
     * @tparam NC_PARAM N cache blocking (0=auto, default 896 fits L2)
     * @tparam KC_PARAM K cache blocking in K-blocks (0=auto, default 128 max storage)
     */
    template <int MR_PARAM = 32, int NR_PARAM = 64, int PREFETCH_A_PARAM = 4, int VECTOR_WIDTH_PARAM = 8,
              int NC_PARAM = 0, int KC_PARAM = 0>
    class Q8_0GemmKernelTemplate
    {
    public:
        static constexpr int MR = MR_PARAM;                     // M register blocking
        static constexpr int NR = NR_PARAM;                     // N register blocking
        static constexpr int BLOCK_SIZE = 32;                   // Q8_0Block::BLOCK_SIZE
        static constexpr int PREFETCH_A = PREFETCH_A_PARAM;     // A prefetch distance
        static constexpr int VECTOR_WIDTH = VECTOR_WIDTH_PARAM; // Post-processing vector width (8=fast, 16=safe)
        static constexpr int NC = NC_PARAM;                     // N cache blocking (0=auto)
        static constexpr int KC = KC_PARAM;                     // K cache blocking in blocks (0=auto)

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

        /**
         * @brief Main GEMM entry point: C = A × B (Q8_0 × Q8_0 → FP32)
         *
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A / rows in B (must be multiple of 32)
         * @param A Q8_0 quantized matrix A (M × K)
         * @param B Q8_0 quantized matrix B (K × N)
         * @param C Output FP32 matrix (M × N)
         * @param ldc Leading dimension of C
         *
         * @note K must be a multiple of BLOCK_SIZE (32)
         * @note C must be pre-allocated (M × N floats)
         */
        static void gemm(int M, int N, int K,
                         const Q8_0Tensor &A,
                         const Q8_0Tensor &B,
                         float *C, int ldc)
        {

            // Validate dimensions
            if (K % BLOCK_SIZE != 0)
            {
                throw std::runtime_error("Q8_0 GEMM: K must be multiple of 32");
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
                        std::cout << "[Q8_0 GEMM] Using " << num_threads << " threads (max=" << max_threads << ")" << std::endl;
                    }
                }
                first_call = false;
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

            // Get raw block data using the method from decode_block_at implementation
            // A and B are structured as: [rows][K_blocks] with Q8_0Block structs
            // We can access blocks via get_raw_block_at() or directly cast from internal storage

            // For now, we'll use a helper to get the first block pointer and compute offsets
            const Q8_0Block *A_blocks = reinterpret_cast<const Q8_0Block *>(
                A.get_raw_block_at(0, 0));
            const Q8_0Block *B_blocks = reinterpret_cast<const Q8_0Block *>(
                B.get_raw_block_at(0, 0));

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

                        // Get A blocks for this row panel [i:i+MR], KC slice
                        const Q8_0Block *A_panel = A_blocks + i * K_blocks + kc_start;

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
                                // Pass K_blocks (original, for A stride) and kc_size (blocks to process)
                                microkernel_full(kc_size, K_blocks, A_panel, B_panels[panel], C_block, ldc);
                            }
                            else
                            {
                                // Edge case: partial tile
                                microkernel_edge(m_block, n_block, kc_size, K_blocks, A_panel,
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
                    std::cout << "[Q8_0 PROFILE] M=" << M << " N=" << N << " K=" << K << std::endl;
                    std::cout << "  B packing:  " << t_pack_ms << " ms (" << (100.0 * t_pack_ms / t_total_ms) << "%)" << std::endl;
                    std::cout << "  GEMM:       " << t_gemm_ms << " ms (" << (100.0 * t_gemm_ms / t_total_ms) << "%)" << std::endl;
                    std::cout << "  Total:      " << t_total_ms << " ms" << std::endl;
                    double gflops = (2.0 * M * N * K) / (t_total_ms * 1e6);
                    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
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
        static void microkernel_full(
            int K_blocks,
            int A_stride,              // NEW: Stride between rows in A (original K, may be > K_blocks with KC blocking)
            const Q8_0Block *A_blocks, // [MR][A_stride] (was [MR][K_blocks])
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
            // sum_a[ir][kb] = sum of A elements for row ir, K-block kb

            auto t_buffer_init_start = std::chrono::high_resolution_clock::now();

            // OPTIMIZATION: Dynamic heap allocation (was thread-local static, but caused stack overflow)
            // With large microkernels (48×48), thread-local static arrays exceeded thread stack limits
            // Heap allocation adds ~1μs overhead but prevents crashes
            // For small microkernels (8×8-16×16), this is still L1-cacheable

            // Allocate buffers dynamically based on actual K_blocks needed
            std::vector<int32_t> accum_vec(MR * NR * K_blocks, 0);
            std::vector<int32_t> sum_a_vec(MR * K_blocks, 0);
            std::vector<uint16_t> a_scales_vec(MR * K_blocks);

            int32_t *accum_storage = accum_vec.data();
            int32_t *sum_a_storage = sum_a_vec.data();
            uint16_t *a_scales_storage = a_scales_vec.data();

            auto t_buffer_init_end = std::chrono::high_resolution_clock::now();

            if (enable_detailed_profiling)
            {
                perf_stats.t_buffer_init_ms += std::chrono::duration<double, std::milli>(t_buffer_init_end - t_buffer_init_start).count();
            }

            // Helper lambdas for 3D indexing
            auto accum = [&](int ir, int jr, int kb) -> int32_t &
            {
                return accum_storage[ir * NR * K_blocks + jr * K_blocks + kb];
            };
            auto sum_a = [&](int ir, int kb) -> int32_t &
            {
                return sum_a_storage[ir * K_blocks + kb];
            };
            auto a_scales = [&](int ir, int kb) -> uint16_t &
            {
                return a_scales_storage[ir * K_blocks + kb];
            };

            const __m512i ones_u8 = _mm512_set1_epi8(1);
            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();

            // Timing accumulators for K-loop phases
            double t_load_a_accum = 0.0;
            double t_load_b_accum = 0.0;
            double t_compute_accum = 0.0;

            // K-loop over blocks
            // NOTE: K-loop unrolling (4×) doesn't help Q8_0 because we need per-block results
            // (each block has different scales, can't accumulate across blocks in registers)
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                auto t_load_a_start = std::chrono::high_resolution_clock::now();

                // OPTIMIZATION: Prefetch A blocks for next iteration
                // Prefetch A blocks ahead (4 blocks empirically optimal)
                // Tested range: 0 (disabled) to 5 blocks ahead
                if (kb + PREFETCH_A < K_blocks)
                {
                    for (int ir = 0; ir < MR; ++ir)
                    {
                        _mm_prefetch(
                            reinterpret_cast<const char *>(&A_blocks[ir * A_stride + kb + PREFETCH_A]),
                            _MM_HINT_T0 // Prefetch to L1 cache
                        );
                    }
                }

                // Load A blocks for 8 rows (8 × 32 int8 = 8 YMM, zero-extended to ZMM)
                __m512i a_vec[MR];
                for (int ir = 0; ir < MR; ++ir)
                {
                    const Q8_0Block &a_block = A_blocks[ir * A_stride + kb];

                    // OPTIMIZATION: Extract scale while block is hot in cache
                    // This eliminates the need for expensive gather in post-processing
                    a_scales(ir, kb) = a_block.d;

                    // Load 32 bytes (256 bits) and zero-extend to 512 bits
                    __m256i a_ymm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(a_block.qs));
                    // Zero-extend to 512 bits (upper 256 bits = 0)
                    a_vec[ir] = _mm512_castsi256_si512(a_ymm);
                }

                auto t_load_a_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_load_a_accum += std::chrono::duration<double, std::milli>(t_load_a_end - t_load_a_start).count();
                }

                auto t_load_b_start = std::chrono::high_resolution_clock::now();

                // Load B blocks for 8 columns - OPTIMIZED: 2 columns per ZMM (no padding!)
                // Layout: [kb][jr_pair][64] where 64 bytes = [col0_32bytes][col1_32bytes]
                __m512i b_vec[NR];
                const uint8_t *B_block_base = B_quants + kb * (NR / 2) * 64;

                for (int jr_pair = 0; jr_pair < NR / 2; ++jr_pair)
                {
                    // Load 64 bytes containing 2 columns
                    const uint8_t *zmm_ptr = B_block_base + jr_pair * 64;
                    __m512i zmm_pair = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(zmm_ptr));

                    // Extract first column (lower 256 bits, zero-extend to 512)
                    __m256i col0_ymm = _mm512_castsi512_si256(zmm_pair);
                    b_vec[jr_pair * 2] = _mm512_castsi256_si512(col0_ymm);

                    // Extract second column (upper 256 bits, zero-extend to 512)
                    __m256i col1_ymm = _mm512_extracti64x4_epi64(zmm_pair, 1);
                    b_vec[jr_pair * 2 + 1] = _mm512_castsi256_si512(col1_ymm);
                }

                auto t_load_b_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_load_b_accum += std::chrono::duration<double, std::milli>(t_load_b_end - t_load_b_start).count();
                }

                auto t_compute_start = std::chrono::high_resolution_clock::now();

                // Compute MR×NR outer product and sum_A accumulation FOR THIS KB
                for (int ir = 0; ir < MR; ++ir)
                {
                    // Accumulate sum_A[ir][kb] for compensation
                    __m512i sum_vec = _mm512_setzero_si512();
                    sum_vec = _mm512_dpbusd_epi32(sum_vec, ones_u8, a_vec[ir]);

                    // Horizontal sum to get scalar sum_A
                    int32_t sum_scalar = _mm512_reduce_add_epi32(sum_vec);
                    sum_a(ir, kb) = sum_scalar;

                    // Inner product with each B column
                    for (int jr = 0; jr < NR; ++jr)
                    {
                        __m512i acc_vec = _mm512_setzero_si512();
                        acc_vec = _mm512_dpbusd_epi32(acc_vec, b_vec[jr], a_vec[ir]);

                        // Horizontal sum to get scalar dot product
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

            auto t_postprocess_start = std::chrono::high_resolution_clock::now();

            // Post-processing: vectorized compensation, scaling, and accumulation to C
            // Correct formula: C[i,j] = Σ_kb (a_scale[kb] * b_scale[kb] * (dot_kb - sum_a[kb] * 128))
            //
            // Vectorization strategy: Process K_blocks in chunks of 16 using AVX-512
            // - Load 16 int32 accumulators → convert to float
            // - Load 16 int32 sum_a values → convert to float
            // - Gather 16 fp16 a_scales (from Q8_0Block.d) → convert to float
            // - Load 16 fp16 b_scales (packed contiguously) → convert to float
            // - Compute: (accum - sum_a * 128) * a_scale * b_scale
            // - Horizontal sum across all K_blocks

            const __m512 compensation_const = _mm512_set1_ps(128.0f);

            // CRITICAL: Use B_packed.K_blocks for B_scales indexing (not K_blocks parameter)
            // With KC blocking, B_packed was created with kc_size < total K_blocks
            // Using wrong stride causes buffer overflow
            const int B_K_blocks = B_packed.K_blocks;

            for (int ir = 0; ir < MR; ++ir)
            {
                for (int jr = 0; jr < NR; ++jr)
                {
                    __m512 result_vec = _mm512_setzero_ps();

                    // Process K_blocks in chunks of VECTOR_WIDTH (template parameter)
                    // VECTOR_WIDTH ∈ {8, 16, 32}: balance ILP vs loop overhead vs register pressure
                    int kb = 0;
                    for (; kb + VECTOR_WIDTH <= K_blocks; kb += VECTOR_WIDTH)
                    {
                        // Load VECTOR_WIDTH int32 accumulators and convert to float
                        __m512i accum_i32 = _mm512_loadu_si512(&accum(ir, jr, kb));
                        __m512 accum_f32 = _mm512_cvtepi32_ps(accum_i32);

                        // Load VECTOR_WIDTH int32 sum_a values and convert to float
                        __m512i sum_a_i32 = _mm512_loadu_si512(&sum_a(ir, kb));
                        __m512 sum_a_f32 = _mm512_cvtepi32_ps(sum_a_i32);

                        // OPTIMIZATION: Load VECTOR_WIDTH fp16 a_scales sequentially (not gather!)
                        // Scales were extracted during K-loop, now contiguous in memory
                        __m256i a_scales_fp16 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(&a_scales(ir, kb)));
                        __m512 a_scales_f32 = _mm512_cvtph_ps(a_scales_fp16);

                        // OPTIMIZATION: Load VECTOR_WIDTH fp16 b_scales sequentially (not gather!)
                        // B scales transposed to [jr][kb] layout for stride-1 access
                        // CRITICAL: Use B_K_blocks (from packed panel) not K_blocks parameter!
                        __m256i b_scales_fp16 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(&B_scales[jr * B_K_blocks + kb]));
                        __m512 b_scales = _mm512_cvtph_ps(b_scales_fp16);

                        // PHASE 2 OPTIMIZATION: Prefetch B scales 4 chunks (64 elements) ahead
                        // Unit stride access pattern makes prefetching more effective than A scales
                        if (kb + 64 < K_blocks)
                        {
                            _mm_prefetch(reinterpret_cast<const char *>(&B_scales[jr * B_K_blocks + kb + 64]), _MM_HINT_T0);
                        }

                        // Compute: compensated = accum - sum_a * 128
                        __m512 compensated = _mm512_fnmadd_ps(sum_a_f32, compensation_const, accum_f32);

                        // Compute: compensated * a_scale * b_scale
                        __m512 scaled = _mm512_mul_ps(compensated, a_scales_f32);
                        scaled = _mm512_mul_ps(scaled, b_scales);

                        // Accumulate to result
                        result_vec = _mm512_add_ps(result_vec, scaled);
                    }

                    // Horizontal sum of VECTOR_WIDTH floats
                    float result = _mm512_reduce_add_ps(result_vec);

                    // Scalar tail for remaining K_blocks (< VECTOR_WIDTH)
                    for (; kb < K_blocks; ++kb)
                    {
                        float a_scale = fp16_to_fp32(a_scales(ir, kb));
                        float b_scale = fp16_to_fp32(B_scales[jr * B_K_blocks + kb]); // Transposed layout, use B_K_blocks!

                        int32_t compensated = accum(ir, jr, kb) - sum_a(ir, kb) * 128;
                        result += static_cast<float>(compensated) * a_scale * b_scale;
                    }

                    C[ir * ldc + jr] = result;
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
                        std::cout << "\n[Q8_0 DETAILED PROFILE - Thread " << thread_id << "]" << std::endl;
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
        }

        /**
         * @brief Edge-case microkernel (partial tiles)
         *
         * Scalar fallback for non-multiple-of-8 dimensions
         */
        static void microkernel_edge(
            int m_block, int n_block, int K_blocks, int A_stride,
            const Q8_0Block *A_blocks,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();
            const int B_K_blocks = B_packed.K_blocks; // Use packed panel's K dimension

            for (int i = 0; i < m_block; ++i)
            {
                for (int j = 0; j < n_block; ++j)
                {
                    int32_t accum = 0;
                    int32_t sum_a = 0;
                    float total_scale = 0.0f;

                    for (int kb = 0; kb < K_blocks; ++kb)
                    {
                        const Q8_0Block &a_block = A_blocks[i * A_stride + kb];

                        // Compute block dot product and sum_A
                        int32_t block_dot = 0;
                        int32_t block_sum_a = 0;

                        const uint8_t *b_col_base = B_quants + (kb * BLOCK_SIZE * NR + j);

                        for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                        {
                            int8_t a_val = a_block.qs[k_in];
                            uint8_t b_val_u8 = b_col_base[k_in * NR];
                            int8_t b_val = static_cast<int8_t>(b_val_u8 - 128);

                            block_dot += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                            block_sum_a += static_cast<int32_t>(a_val);
                        }

                        accum += block_dot;
                        sum_a += block_sum_a;

                        // Accumulate scale (using transposed B scale layout)
                        float a_scale = fp16_to_fp32(a_block.d);
                        float b_scale = fp16_to_fp32(B_scales[j * B_K_blocks + kb]); // Use B_K_blocks!
                        total_scale += a_scale * b_scale;
                    }

                    // Compensation
                    int32_t compensated = accum - sum_a * 128;

                    // Final result
                    float result = static_cast<float>(compensated) * (total_scale / K_blocks);
                    C[i * ldc + j] += result;
                }
            }
        }
    };

    // Type aliases for common microkernel sizes
    // 32×64 with prefetch=4, vector_width=8 is optimal (78% speedup over 8×8)
    // Default parameters optimized via comprehensive benchmark sweep
    // See tests/v2/performance/Perf__Q8_0Gemm_MicrokernelSize.cpp for full results
    using Q8_0GemmKernel = Q8_0GemmKernelTemplate<32, 64, 4, 8>;        // Default (optimal: 377 GFLOPS, 78% vs 8×8)
    using Q8_0GemmKernel_8x8 = Q8_0GemmKernelTemplate<8, 8, 2, 16>;     // Original baseline (212 GFLOPS)
    using Q8_0GemmKernel_16x16 = Q8_0GemmKernelTemplate<16, 16, 2, 16>; // Previous default (278 GFLOPS, 31%)
    using Q8_0GemmKernel_32x32 = Q8_0GemmKernelTemplate<32, 32, 2, 16>; // Best square (295 GFLOPS, 39%)

} // namespace llaminar2
