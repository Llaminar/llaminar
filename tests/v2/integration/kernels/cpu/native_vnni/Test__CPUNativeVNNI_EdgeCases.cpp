/**
 * @file Test__CPUNativeVNNI_EdgeCases.cpp
 * @brief Edge case and boundary condition tests for CPUNativeVNNI GEMM/GEMV.
 *
 * Probes alignment boundaries, tail chunk handling, and unusual matrix sizes
 * that could trigger segfaults or buffer overflows in the AVX-512 kernels.
 *
 * Key boundary conditions tested:
 *   N-dimension: chunk boundary (64), sub-chunk (1..63), just-over (65..127)
 *   K-dimension: block boundary (32), sub-block (1..31), just-over (33..63)
 *   M-dimension: odd (1,3,5,7), even (2,4,8), edge (1)
 *   Combined: double-tail (N%64!=0 && K%32!=0), minimal (N=1,K=32)
 *
 * @note Run with:  ctest -R V2_Integration_CPUNativeVNNI_EdgeCases
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "fort.hpp"

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // MPI global environment: init once, finalize on exit, abort on crash
    // =========================================================================
    void mpi_abort_signal_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal caught in edge-case test — calling MPI_Abort\n";
        [[maybe_unused]] auto _ = write(STDERR_FILENO, msg, strlen(msg));
        MPI_Abort(MPI_COMM_WORLD, sig);
        _exit(128 + sig);
    }

    class MPIEnvironment : public ::testing::Environment
    {
    public:
        void SetUp() override
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                MPI_Init(nullptr, nullptr);
            std::signal(SIGSEGV, mpi_abort_signal_handler);
            std::signal(SIGABRT, mpi_abort_signal_handler);
            std::signal(SIGFPE, mpi_abort_signal_handler);
        }
        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };

    static auto *g_mpi_env [[maybe_unused]] =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    // =========================================================================
    // FP32 reference (double-precision accumulation)
    // =========================================================================

    void cpuFP32GemvReference(const TensorBase *weights, const float *A,
                              float *C, int N, int K)
    {
        const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        ASSERT_NE(unpackable, nullptr);

        const int K_blocks = (K + 31) / 32;

        for (int n = 0; n < N; ++n)
        {
            double acc = 0.0;
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                int8_t vals[32];
                unpackable->unpack_block_to_int8(n, kb, vals);
                float scale = unpackable->get_block_scale(n, kb);
                float min_val = unpackable->get_block_min(n, kb);

                for (int i = 0; i < 32; ++i)
                {
                    int k_idx = kb * 32 + i;
                    if (k_idx >= K)
                        break;
                    double fp_w = static_cast<double>(scale) * static_cast<double>(vals[i]) + static_cast<double>(min_val);
                    acc += fp_w * static_cast<double>(A[k_idx]);
                }
            }
            C[n] = static_cast<float>(acc);
        }
    }

    void cpuFP32GemmReference(const TensorBase *weights, const float *A,
                              float *C, int M, int N, int K)
    {
        for (int m = 0; m < M; ++m)
            cpuFP32GemvReference(weights, A + m * K, C + m * N, N, K);
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    float cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0, norm_a = 0, norm_b = 0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += (double)a[i] * (double)b[i];
            norm_a += (double)a[i] * (double)a[i];
            norm_b += (double)b[i] * (double)b[i];
        }
        if (norm_a < 1e-15 || norm_b < 1e-15)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    // =========================================================================
    // Weight factory (key formats covering all decode paths)
    // =========================================================================

    std::unique_ptr<TensorBase> createWeightsForFormat(
        const std::string &fmt_name, size_t N, size_t K)
    {
        if (fmt_name == "Q4_0")
            return TestTensorFactory::createQ4_0Random({N, K});
        if (fmt_name == "IQ4_NL")
            return TestTensorFactory::createIQ4_NLRandom({N, K});
        if (fmt_name == "Q4_1")
            return TestTensorFactory::createQ4_1Random({N, K});
        if (fmt_name == "Q5_0")
            return TestTensorFactory::createQ5_0Random({N, K});
        if (fmt_name == "Q5_1")
            return TestTensorFactory::createQ5_1Random({N, K});
        if (fmt_name == "Q6_K")
            return TestTensorFactory::createQ6_KRandom({N, K});
        if (fmt_name == "Q3_K")
            return TestTensorFactory::createQ3_KRandom({N, K});
        if (fmt_name == "Q8_0")
            return TestTensorFactory::createQ8_0Random({N, K});
        return nullptr;
    }

    // Formats covering all decode paths: nibble-LUT, asymmetric, INT8-perblock,
    // INT8-superblock, and 8-bit direct
    struct FormatSpec
    {
        std::string name;
        float cosine_threshold;
    };

    static const std::vector<FormatSpec> EDGE_FORMATS = {
        {"Q4_0", 0.985f}, // nibble-LUT symmetric
        {"Q4_1", 0.985f}, // nibble-LUT asymmetric (has min correction)
        {"Q5_0", 0.985f}, // INT8 pre-decoded per-block
        {"Q6_K", 0.985f}, // INT8 pre-decoded superblock
        {"Q3_K", 0.970f}, // INT8 superblock (low-bit, lower threshold)
        {"Q8_0", 0.998f}, // INT8 direct (highest accuracy)
    };

    // =========================================================================
    // Test fixture
    // =========================================================================

    class CPUNativeVNNIEdgeCaseTest : public ::testing::Test
    {
    protected:
        struct TestResult
        {
            std::string fmt;
            int M, N, K;
            float min_cosine;
            bool passed;
            bool crashed; // Set if kernel returns false
        };

        /**
         * @brief Run a single GEMM/GEMV correctness test.
         * @return TestResult with cosine similarity and pass/fail status.
         */
        TestResult runTest(const std::string &fmt, int M, int N, int K, float threshold)
        {
            TestResult res{fmt, M, N, K, 0.0f, false, false};

            auto weights = createWeightsForFormat(fmt, N, K);
            if (!weights)
            {
                ADD_FAILURE() << fmt << " weight creation failed for " << N << "x" << K;
                res.crashed = true;
                return res;
            }

            CPUNativeVNNIGemmKernel kernel(weights.get());
            if (!kernel.isValid())
            {
                ADD_FAILURE() << fmt << " pack failed [" << N << "x" << K << "]";
                res.crashed = true;
                return res;
            }

            std::vector<float> A(static_cast<size_t>(M) * K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            // Allocate output with exact size — no extra padding
            // This catches any kernel writes past the end of the buffer
            std::vector<float> C_native(static_cast<size_t>(M) * N, 0.0f);

            bool ok = kernel.multiply(A.data(), C_native.data(), M, N, K);
            if (!ok)
            {
                ADD_FAILURE() << fmt << " multiply returned false for M=" << M
                              << " N=" << N << " K=" << K;
                res.crashed = true;
                return res;
            }

            // Check for NaN/Inf in output
            for (size_t i = 0; i < C_native.size(); ++i)
            {
                EXPECT_FALSE(std::isnan(C_native[i]))
                    << fmt << " NaN at index " << i << " (M=" << M
                    << " N=" << N << " K=" << K << ")";
                EXPECT_FALSE(std::isinf(C_native[i]))
                    << fmt << " Inf at index " << i << " (M=" << M
                    << " N=" << N << " K=" << K << ")";
                if (std::isnan(C_native[i]) || std::isinf(C_native[i]))
                {
                    res.crashed = true;
                    return res;
                }
            }

            // Compute reference
            std::vector<float> C_ref(static_cast<size_t>(M) * N, 0.0f);
            cpuFP32GemmReference(weights.get(), A.data(), C_ref.data(), M, N, K);

            // Per-row cosine similarity
            float min_cos = 1.0f;
            for (int m = 0; m < M; ++m)
            {
                float cos_sim = cosineSimilarity(
                    C_native.data() + m * N, C_ref.data() + m * N, N);
                min_cos = std::min(min_cos, cos_sim);
            }
            res.min_cosine = min_cos;
            res.passed = min_cos >= threshold;
            EXPECT_GE(min_cos, threshold)
                << fmt << " M=" << M << " N=" << N << " K=" << K
                << " min_cosine=" << min_cos;
            return res;
        }

        /**
         * @brief Run a matrix of (M, N, K) shapes across all edge formats.
         * Prints a summary table. Returns total pass/fail count.
         */
        std::pair<int, int> runEdgeSweep(
            const std::string &test_name,
            const std::vector<std::tuple<int, int, int>> &shapes)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header << "Format" << "M" << "N" << "K"
                  << "Min Cosine" << "Status" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(4).set_cell_text_align(fort::text_align::right);
            table.column(5).set_cell_text_align(fort::text_align::center);

            int pass = 0, total = 0;

            for (const auto &fmt : EDGE_FORMATS)
            {
                for (const auto &[M, N, K] : shapes)
                {
                    auto res = runTest(fmt.name, M, N, K, fmt.cosine_threshold);
                    total++;
                    if (res.passed)
                        pass++;

                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.6f", res.min_cosine);
                    const char *status = res.crashed  ? "\xe2\x9c\x97 CRASH"
                                         : res.passed ? "\xe2\x9c\x93"
                                                      : "\xe2\x9c\x97";
                    table << fmt.name << M << N << K << buf << status << fort::endr;
                }
                table << fort::separator;
            }

            table << "TOTAL" << "" << "" << "" << ""
                  << std::to_string(pass) + "/" + std::to_string(total) << fort::endr;

            std::cout << "\n=== " << test_name << " ===\n"
                      << table.to_string() << std::endl;

            return {pass, total};
        }
    };

    // =========================================================================
    // TEST 1: N-dimension tail chunks (N not a multiple of 64)
    //
    // The AVX-512 kernels write 64 floats per chunk. When N%64 != 0,
    // a tail chunk with fewer columns must use a temp buffer to avoid
    // overflowing the output array.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMV_N_TailChunks)
    {
        // M=1 (GEMV path), K=128 (aligned)
        // N values probe: sub-chunk, on-boundary, just-over, 1.5-chunk, 2-chunks-1
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 1, 128},   // N=1:   1 chunk, 1 valid column
            {1, 2, 128},   // N=2:   1 chunk, 2 valid columns
            {1, 15, 128},  // N=15:  1 chunk, 15 valid columns
            {1, 31, 128},  // N=31:  1 chunk, 31 valid columns
            {1, 32, 128},  // N=32:  1 chunk, 32 valid columns
            {1, 33, 128},  // N=33:  1 chunk, 33 valid columns
            {1, 63, 128},  // N=63:  1 chunk, 63 valid columns (max tail)
            {1, 64, 128},  // N=64:  exact 1 chunk, no tail
            {1, 65, 128},  // N=65:  2 chunks, chunk[1] has 1 col
            {1, 96, 128},  // N=96:  2 chunks, chunk[1] has 32 cols
            {1, 97, 128},  // N=97:  2 chunks, chunk[1] has 33 cols
            {1, 127, 128}, // N=127: 2 chunks, chunk[1] has 63 cols
            {1, 128, 128}, // N=128: exact 2 chunks
            {1, 129, 128}, // N=129: 3 chunks, chunk[2] has 1 col
        };
        auto [pass, total] = runEdgeSweep("GEMV N-Dimension Tail Chunks", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 2: K-dimension tail blocks (K not a multiple of 32)
    //
    // Quantization pads the last K-block to 32 elements. Verifies the
    // padding logic produces correct results, not garbage/NaN.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMV_K_BlockCounts)
    {
        // M=1 (GEMV path), N=128 (aligned)
        // K must be a multiple of 32 (quantization block size).
        // Tests varying block counts to exercise different K-parallel thresholds.
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 128, 32},  // K=32:  1 block (minimum)
            {1, 128, 64},  // K=64:  2 blocks
            {1, 128, 96},  // K=96:  3 blocks
            {1, 128, 128}, // K=128: 4 blocks
            {1, 128, 160}, // K=160: 5 blocks
            {1, 128, 192}, // K=192: 6 blocks
            {1, 128, 224}, // K=224: 7 blocks (Wo_proj TP=4)
            {1, 128, 256}, // K=256: 8 blocks
            {1, 128, 512}, // K=512: 16 blocks
            {1, 128, 896}, // K=896: 28 blocks (Qwen 0.5B)
        };
        auto [pass, total] = runEdgeSweep("GEMV K-Dimension Block Counts", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 3: Double-tail (both N%64!=0 and K%32!=0)
    //
    // Most pathological case: tail handling needed on both dimensions.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMV_N_Tail_K_Varied)
    {
        // N is unaligned (not multiple of 64), K varies (must be multiple of 32).
        // Tests interaction between N-tail chunk handling and different K sizes.
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 1, 32},    // N=1, K=1 block
            {1, 15, 32},   // N=15, K=1 block
            {1, 31, 64},   // N=31, K=2 blocks
            {1, 33, 64},   // N=33, K=2 blocks
            {1, 63, 96},   // N=63 (max tail), K=3 blocks
            {1, 65, 32},   // N=65 (2 chunks, 1-col tail), K=1 block
            {1, 97, 128},  // N=97 (2 chunks, 33-col tail), K=4 blocks
            {1, 127, 256}, // N=127 (2 chunks, 63-col tail), K=8 blocks
            {1, 129, 64},  // N=129 (3 chunks, 1-col tail), K=2 blocks
            {1, 33, 896},  // N=33, K=Qwen 0.5B dim
            {1, 97, 896},  // N=97, K=Qwen 0.5B dim
        };
        auto [pass, total] = runEdgeSweep("GEMV N-Tail + K-Varied", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 4: GEMM N-dimension tail chunks (M>1, 2-row microkernel)
    //
    // The 2-row microkernel writes to 2 rows simultaneously. When N%64!=0,
    // tail chunk handling must work for both rows without overflow.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_N_TailChunks)
    {
        std::vector<std::tuple<int, int, int>> shapes = {
            {2, 1, 128},   // N=1, exact 2-row
            {2, 31, 128},  // N=31, exact 2-row, max single-chunk tail
            {2, 32, 128},  // N=32 < 64, 2-row
            {2, 33, 128},  // N=33, 2-row
            {2, 63, 128},  // N=63 max tail, 2-row
            {2, 64, 128},  // N=64 exact chunk, 2-row
            {2, 65, 128},  // N=65, 2 chunks (1 col tail), 2-row
            {2, 127, 128}, // N=127, 2 chunks (63 col tail), 2-row
            {2, 128, 128}, // N=128, exact 2 chunks, 2-row
            {3, 33, 128},  // N=33, 2-row + 1-row remainder
            {3, 65, 128},  // N=65, 2-row + 1-row, 2 chunks
            {3, 127, 128}, // N=127, 2-row + 1-row, chunk tail
            {4, 33, 128},  // N=33, 2×2-row
            {4, 65, 128},  // N=65, 2×2-row, 2 chunks
            {5, 33, 128},  // N=33, 2×2-row + 1-row remainder
            {5, 65, 128},  // N=65, 2×2-row + 1-row, 2 chunks
            {7, 97, 128},  // N=97, 3×2-row + 1-row, 2 chunks (33 col tail)
        };
        auto [pass, total] = runEdgeSweep("GEMM N-Tail Chunks (2-row microkernel)", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 5: GEMM K-dimension tail blocks (M>1)
    //
    // K-tail requires correct quantization padding for all M rows.
    // Also exercises K-tiling when K is large enough.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_K_BlockCounts)
    {
        // K must be multiple of 32. Tests different block counts with M>1.
        std::vector<std::tuple<int, int, int>> shapes = {
            {2, 128, 32},  // K=1 block, 2-row
            {2, 128, 64},  // K=2 blocks, 2-row
            {2, 128, 96},  // K=3 blocks, 2-row
            {3, 128, 32},  // K=1 block, 2-row + 1-row
            {3, 128, 128}, // K=4 blocks, 2-row + 1-row
            {4, 128, 64},  // K=2 blocks, 2×2-row
            {4, 128, 224}, // K=7 blocks (Wo TP=4)
            {8, 128, 128}, // K=4 blocks, 4×2-row
            {8, 128, 896}, // K=28 blocks (Qwen 0.5B)
        };
        auto [pass, total] = runEdgeSweep("GEMM K-Block Counts (2-row microkernel)", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 6: GEMM double-tail (both N and K unaligned, M>1)
    //
    // Most pathological GEMM case: tail handling on N in the 2-row microkernel
    // AND K-tail in activation quantization, plus odd M for remainder row.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_N_Tail_K_Varied)
    {
        // N unaligned + K varies (multiples of 32) + various M values.
        // Combines N-tail chunk handling with different K block counts and odd M.
        std::vector<std::tuple<int, int, int>> shapes = {
            {2, 33, 32},   // N-tail, K=1 block, 2-row
            {3, 33, 64},   // N-tail, K=2 blocks, odd M
            {2, 65, 64},   // 2 N-chunks, 1-col tail, 2-row
            {3, 65, 96},   // 2 N-chunks, 1-col tail, odd M
            {5, 97, 128},  // 2 N-chunks, 33-col tail, 2×2-row + 1-row
            {7, 129, 256}, // 3 N-chunks, 1-col tail, 3×2-row + 1-row
            {4, 31, 32},   // Sub-chunk, K=1 block, 2×2-row
            {3, 1, 64},    // N=1, odd M
            {2, 63, 96},   // Max N-tail, K=3 blocks, 2-row
            {8, 33, 896},  // N-tail, K=Qwen 0.5B dim, 4×2-row
        };
        auto [pass, total] = runEdgeSweep("GEMM N-Tail + K-Varied + Odd M", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 7: Minimal matrix sizes (stress minimum dimensions)
    //
    // Tests the smallest possible matrices that the kernel should support.
    // N=1,K=32 is the absolute minimum for a single-element output.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, MinimalMatrices)
    {
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 1, 32},  // 1×1 output: GEMV, N=1, K=1 block
            {1, 1, 64},  // 1×1 output: GEMV, N=1, K=2 blocks
            {1, 2, 32},  // 1×2 output: GEMV
            {1, 32, 32}, // 32×32 weight matrix: GEMV, single sub-chunk, single block
            {2, 1, 32},  // 2×1 output: GEMM 2-row, minimal N and K
            {2, 2, 32},  // 2×2 output: GEMM 2-row
            {2, 32, 32}, // 2×32: GEMM 2-row, sub-chunk
            {3, 1, 32},  // 3×1: GEMM 2-row + 1-row, minimal everything
            {1, 64, 32}, // Single full chunk, single block: ideal alignment
        };
        auto [pass, total] = runEdgeSweep("Minimal Matrix Sizes", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 8: Power-of-two aligned shapes (sanity baseline)
    //
    // These shapes should always work perfectly. Used as a regression
    // baseline to detect if the edge-case fixes broke aligned paths.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, AlignedBaseline)
    {
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 64, 32},   // Smallest perfect GEMV
            {1, 64, 64},   // 1 chunk, 2 blocks
            {1, 128, 64},  // 2 chunks, 2 blocks
            {1, 128, 128}, // 2 chunks, 4 blocks
            {1, 256, 256}, // Standard small matrix
            {1, 256, 896}, // Qwen 0.5B attention
            {2, 64, 64},   // Smallest perfect GEMM 2-row
            {2, 128, 128}, // 2 chunks, 4 blocks, 2-row
            {4, 256, 256}, // Standard GEMM
            {8, 256, 256}, // Multi-row GEMM
        };
        auto [pass, total] = runEdgeSweep("Aligned Baseline (power-of-2)", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 9: Small-N 2D dispatch boundary
    //
    // The GEMM path activates small-N 2D dispatch when
    // total_n_blocks <= num_threads/4 && M >= 2.
    // This creates an M×N task grid instead of N-only parallelism.
    // With 56 threads, threshold is ~14 N-blocks → N <= 896.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_SmallN_2DDispatch)
    {
        const int K = 256;
        std::vector<std::tuple<int, int, int>> shapes = {
            // These should trigger 2D dispatch (small N, M>=2)
            {2, 32, K},  // 1 N-chunk → 2D dispatch
            {4, 32, K},  // 1 N-chunk, more M rows
            {8, 64, K},  // 1 N-chunk, 8 M rows
            {16, 32, K}, // 1 N-chunk, 16 M rows → lots of M-parallelism
            {2, 128, K}, // 2 N-chunks
            {8, 128, K}, // 2 N-chunks, 8 rows

            // These should use N-only dispatch (large N or M=1)
            {1, 32, K},   // M=1 → GEMV path
            {2, 896, K},  // Many N-chunks
            {1, 4864, K}, // FFN-sized N, GEMV

            // Tail chunks in 2D dispatch
            {4, 33, K}, // 1 N-chunk with tail, 2D dispatch
            {8, 65, K}, // 2 N-chunks (1 col tail), 2D dispatch
            {3, 97, K}, // 2 N-chunks (33 col tail), odd M, 2D dispatch
        };
        auto [pass, total] = runEdgeSweep("GEMM Small-N 2D Dispatch", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 10: Odd M values sweep (2-row microkernel remainder)
    //
    // The 2-row microkernel processes rows in pairs. Odd M values
    // always produce a single remainder row that must use the 1-row
    // fallback without corrupting other rows.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_OddM_Sweep)
    {
        const int N = 256, K = 256;
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, N, K},  // M=1: GEMV, no 2-row at all
            {2, N, K},  // M=2: exactly 1 pair, no remainder
            {3, N, K},  // M=3: 1 pair + 1 remainder
            {4, N, K},  // M=4: 2 pairs, no remainder
            {5, N, K},  // M=5: 2 pairs + 1 remainder
            {6, N, K},  // M=6: 3 pairs, no remainder
            {7, N, K},  // M=7: 3 pairs + 1 remainder
            {8, N, K},  // M=8: 4 pairs, no remainder
            {9, N, K},  // M=9: 4 pairs + 1 remainder
            {15, N, K}, // M=15: 7 pairs + 1 remainder
            {16, N, K}, // M=16: 8 pairs
            {17, N, K}, // M=17: 8 pairs + 1 remainder
        };
        auto [pass, total] = runEdgeSweep("GEMM Odd-M Sweep (2-row microkernel)", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 11: TP-split realistic shapes
    //
    // When tensor-parallel shards reduce N or K, we get shapes that don't
    // appear in the original model. These test the actual TP-generated dims.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMV_TP_RealisticShapes)
    {
        // Qwen 0.5B TP-split shapes (GEMV M=1)
        std::vector<std::tuple<int, int, int>> shapes = {
            // TP=2 Column-parallel (splits N)
            {1, 448, 896},  // Q_proj TP=2
            {1, 64, 896},   // K/V_proj TP=2
            {1, 2432, 896}, // FFN_Gate/Up TP=2

            // TP=4 Column-parallel
            {1, 224, 896},  // Q_proj TP=4
            {1, 32, 896},   // K/V_proj TP=4  (only 1 chunk, 32 cols!)
            {1, 1216, 896}, // FFN_Gate/Up TP=4

            // TP=2 Row-parallel (splits K)
            {1, 896, 448},  // Wo_proj TP=2
            {1, 896, 2432}, // FFN_Down TP=2

            // TP=4 Row-parallel
            {1, 896, 224},  // Wo_proj TP=4
            {1, 896, 1216}, // FFN_Down TP=4
        };
        auto [pass, total] = runEdgeSweep("GEMV TP-Realistic Shapes (0.5B)", shapes);
        EXPECT_EQ(pass, total);
    }

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_TP_RealisticShapes)
    {
        // Qwen 0.5B TP-split shapes, M=8 (GEMM prefill path)
        std::vector<std::tuple<int, int, int>> shapes = {
            // TP=4 Column-parallel (small N, triggers 2D dispatch)
            {8, 32, 896},   // K/V_proj TP=4
            {8, 224, 896},  // Q_proj TP=4
            {8, 1216, 896}, // FFN_Gate/Up TP=4

            // TP=4 Row-parallel (small K)
            {8, 896, 224},  // Wo_proj TP=4
            {8, 896, 1216}, // FFN_Down TP=4

            // Odd M + TP shapes
            {3, 32, 896},  // K/V_proj TP=4, odd M
            {5, 224, 896}, // Q_proj TP=4, odd M
            {7, 896, 224}, // Wo_proj TP=4, odd M
        };
        auto [pass, total] = runEdgeSweep("GEMM TP-Realistic Shapes (0.5B, M=8)", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 12: Exact-boundary transitions
    //
    // Values that are exactly on chunk (64) or block (32) boundaries.
    // These test the boundary condition: n_cols == 64 means "full chunk"
    // and must NOT use the temp buffer path.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMV_ExactBoundaries)
    {
        std::vector<std::tuple<int, int, int>> shapes = {
            // Exactly at N-chunk boundaries
            {1, 64, 256},  // 1 chunk
            {1, 128, 256}, // 2 chunks
            {1, 192, 256}, // 3 chunks
            {1, 256, 256}, // 4 chunks
            {1, 320, 256}, // 5 chunks
            {1, 384, 256}, // 6 chunks
            {1, 448, 256}, // 7 chunks  (Q_proj TP=2 for 0.5B)
            {1, 512, 256}, // 8 chunks
            {1, 576, 256}, // 9 chunks
            {1, 640, 256}, // 10 chunks
            {1, 896, 256}, // 14 chunks (Q_proj for 0.5B)

            // Exactly at K-block boundaries
            {1, 128, 32},  // 1 block
            {1, 128, 64},  // 2 blocks
            {1, 128, 96},  // 3 blocks
            {1, 128, 128}, // 4 blocks
            {1, 128, 256}, // 8 blocks
            {1, 128, 512}, // 16 blocks
            {1, 128, 896}, // 28 blocks (Qwen 0.5B K dim)
        };
        auto [pass, total] = runEdgeSweep("GEMV Exact Boundary Values", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 13: Large M stress test
    //
    // Tests with large M values to exercise many iterations of the 2-row
    // microkernel loop and verify pre-quantize barrier works for many rows.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, GEMM_LargeM)
    {
        std::vector<std::tuple<int, int, int>> shapes = {
            {32, 128, 128},  // 16 pairs
            {32, 65, 128},   // 16 pairs, N-tail
            {33, 128, 128},  // 16 pairs + 1 remainder
            {33, 65, 128},   // 16 pairs + 1 remainder, N-tail
            {64, 128, 128},  // 32 pairs
            {64, 97, 128},   // 32 pairs, N-tail
            {128, 128, 128}, // 64 pairs
            {128, 65, 64},   // 64 pairs, N-tail, K=2 blocks
        };
        auto [pass, total] = runEdgeSweep("GEMM Large M Stress", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 14: Primes and awkward sizes
    //
    // Nobody designs models with prime dimensions, but they could appear in
    // custom architectures or TP splits. These test the most "uncooperative"
    // dimensions for SIMD alignment.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, PrimeDimensions)
    {
        // N can be any value (prime); K must be a multiple of 32.
        // Tests prime N values (most "uncooperative" for 64-wide SIMD alignment).
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 37, 32},   // Prime N, K=1 block
            {1, 67, 64},   // Prime N near chunk boundary, K=2 blocks
            {1, 127, 128}, // Prime N near 2-chunk, K=4 blocks
            {1, 251, 256}, // Prime N near power-of-2, K=8 blocks
            {3, 37, 32},   // Prime N, odd M
            {7, 127, 128}, // Prime N, odd M, multiple 2-row iterations
            {5, 67, 64},   // Prime N, 2×2-row + 1-row
            {1, 13, 32},   // Small prime N
            {1, 509, 256}, // Large prime N
            {2, 97, 224},  // Prime N, 2-row, K=7 blocks
        };
        auto [pass, total] = runEdgeSweep("Prime N Dimensions", shapes);
        EXPECT_EQ(pass, total);
    }

    // =========================================================================
    // TEST 15: Single-row weights (N=1) — one output element per M-row
    //
    // N=1 means only 1 output column. The kernel must produce a single
    // float per input row without overwriting adjacent memory.
    // =========================================================================

    TEST_F(CPUNativeVNNIEdgeCaseTest, SingleColumnOutput)
    {
        std::vector<std::tuple<int, int, int>> shapes = {
            {1, 1, 32},  // 1×1, GEMV
            {1, 1, 64},  // 1×1, GEMV, 2 K-blocks
            {1, 1, 128}, // 1×1, GEMV, 4 K-blocks
            {1, 1, 256}, // 1×1, GEMV, 8 K-blocks
            {2, 1, 32},  // 2×1, GEMM 2-row
            {2, 1, 64},  // 2×1, GEMM 2-row, 2 K-blocks
            {2, 1, 128}, // 2×1, GEMM 2-row, 4 K-blocks
            {3, 1, 64},  // 3×1, 2-row + 1-row
            {4, 1, 128}, // 4×1, 2×2-row
            {5, 1, 256}, // 5×1, 2×2-row + 1-row
        };
        auto [pass, total] = runEdgeSweep("Single Column Output (N=1)", shapes);
        EXPECT_EQ(pass, total);
    }

} // namespace
