/**
 * @file Test__AVX2VNNIParity.cpp
 * @brief Parity tests proving AVX2 emulated VNNI produces identical results
 *        to native AVX512-VNNI for all GEMV/GEMM kernel paths.
 *
 * Test levels:
 *  1. Intrinsic-level:  avx2_dpbusd_epi32 vs _mm512_dpbusd_epi32
 *  2. Single-chunk GEMV: 64-column chunk output comparison
 *  3. Full GEMV (M=1):  gemv_native_vnni_preq with ISAPath::AVX512 vs AVX2
 *  4. Full GEMM (M>1):  gemm_native_vnni_preq with ISAPath::AVX512 vs AVX2
 *
 * All tests run on AVX512-VNNI hardware, calling both paths explicitly
 * via the ISAPath runtime dispatch enum.
 */

#include <gtest/gtest.h>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <cmath>

#include "kernels/cpu/gemm/VNNIEmulation.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::cpu::native_vnni::isa;
using namespace llaminar2;
using llaminar2::test::TestTensorFactory;

/**
 * @brief Prove native scale conversion preserves every binary16 bit contract.
 *
 * Finite values exercise the F16C hot path. Infinity and NaN encodings
 * exercise the portable special-value path, including signaling-NaN payloads.
 * Comparing the resulting FP32 bytes rather than floating-point equality also
 * covers signed zero and every NaN payload.
 */
TEST(NativeVNNIFP16Scale, ExhaustiveBinary16ExpansionMatchesPortableBytes)
{
    for (uint32_t raw = 0; raw <= 0xffffu; ++raw)
    {
        const uint16_t fp16 = static_cast<uint16_t>(raw);
        const float expected = llaminar2::fp16_to_fp32(fp16);
        const float actual = nativeVNNIFP16ScaleToFP32(fp16);
        uint32_t expected_bits = 0;
        uint32_t actual_bits = 0;
        std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
        std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
        ASSERT_EQ(actual_bits, expected_bits)
            << "binary16 encoding 0x" << std::hex << raw;
    }
}

// ============================================================================
// Helper functions (free functions for macro accessibility)
// ============================================================================

namespace avx2_parity_helpers
{
    // Create random Q8_1 blocks for activation vectors
    inline std::vector<Q8_1Block> createRandomQ8_1(int K, uint32_t seed = 42)
    {
        int K_blocks = (K + 31) / 32;
        std::vector<Q8_1Block> blocks(K_blocks);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.5f);

        for (int kb = 0; kb < K_blocks; ++kb)
        {
            float scale = scale_dist(rng);
            blocks[kb].d = simd::fp32_to_fp16(scale);
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                blocks[kb].qs[i] = static_cast<int8_t>(dist(rng));
                sum += blocks[kb].qs[i];
            }
            blocks[kb].sum_qs = static_cast<int16_t>(std::clamp(sum, -32768, 32767));
        }
        return blocks;
    }

    // Pack weights and assert success
    inline CPUNativeVNNIPackedWeights packWeights(const TensorBase *tensor)
    {
        CPUNativeVNNIPackedWeights packed;
        bool ok = packWeightsCPUNativeVNNI(tensor, packed);
        if (!ok)
            throw std::runtime_error("Weight packing failed");
        return packed;
    }

    // Compare two float arrays for exact equality
    inline void assertExactEqual(const float *a, const float *b, int n,
                                 const std::string &label)
    {
        float max_diff = 0.0f;
        int max_idx = -1;
        int mismatches = 0;
        for (int i = 0; i < n; ++i)
        {
            float diff = std::fabs(a[i] - b[i]);
            if (diff > 0.0f)
            {
                mismatches++;
                if (diff > max_diff)
                {
                    max_diff = diff;
                    max_idx = i;
                }
            }
        }
        EXPECT_EQ(mismatches, 0)
            << label << ": " << mismatches << "/" << n
            << " mismatches, max diff=" << max_diff
            << " at index " << max_idx
            << (max_idx >= 0 ? (" (avx512=" + std::to_string(a[max_idx]) +
                                " avx2=" + std::to_string(b[max_idx]) + ")")
                             : "");
    }

    inline double relativeL2(const float *expected, const float *actual, int n)
    {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double e = static_cast<double>(expected[i]);
            const double d = e - static_cast<double>(actual[i]);
            num += d * d;
            den += e * e;
        }
        return std::sqrt(num / std::max(den, 1.0e-30));
    }

    inline double cosineSimilarity(const float *a, const float *b, int n)
    {
        double dot = 0.0;
        double aa = 0.0;
        double bb = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double av = static_cast<double>(a[i]);
            const double bv = static_cast<double>(b[i]);
            dot += av * bv;
            aa += av * av;
            bb += bv * bv;
        }
        return dot / std::sqrt(std::max(aa * bb, 1.0e-30));
    }

    inline double symmetricKLFromLogits(const float *a, const float *b, int n)
    {
        auto accumulate = [n](const float *p_logits, const float *q_logits)
        {
            const double max_p = *std::max_element(p_logits, p_logits + n);
            const double max_q = *std::max_element(q_logits, q_logits + n);
            double sum_p = 0.0;
            double sum_q = 0.0;
            for (int i = 0; i < n; ++i)
            {
                sum_p += std::exp(static_cast<double>(p_logits[i]) - max_p);
                sum_q += std::exp(static_cast<double>(q_logits[i]) - max_q);
            }

            double kl = 0.0;
            for (int i = 0; i < n; ++i)
            {
                const double log_p = static_cast<double>(p_logits[i]) - max_p - std::log(sum_p);
                const double log_q = static_cast<double>(q_logits[i]) - max_q - std::log(sum_q);
                const double p = std::exp(log_p);
                kl += p * (log_p - log_q);
            }
            return kl;
        };

        return 0.5 * (accumulate(a, b) + accumulate(b, a));
    }

    inline void assertStrictMetricClose(
        const float *expected,
        const float *actual,
        int n,
        const std::string &label)
    {
        const double rel_l2 = relativeL2(expected, actual, n);
        const double cos = cosineSimilarity(expected, actual, n);
        const double skl = symmetricKLFromLogits(expected, actual, n);

        EXPECT_LE(rel_l2, 1.0e-4) << label << " relative L2";
        EXPECT_GE(cos, 0.999999) << label << " cosine";
        EXPECT_LE(skl, 1.0e-7) << label << " symmetric KL";
    }

    /**
     * @brief Restore process-global OpenMP controls after a thread sweep.
     *
     * GoogleTest fatal assertions unwind the test body, so an RAII guard is
     * required to prevent one failed thread-totality cell from contaminating
     * every later CPU unit test in the process.
     */
    class ScopedOpenMPControls final
    {
    public:
        ScopedOpenMPControls()
            : original_threads_(omp_get_max_threads()),
              original_dynamic_(omp_get_dynamic())
        {
            omp_set_dynamic(0);
        }

        ScopedOpenMPControls(const ScopedOpenMPControls &) = delete;
        ScopedOpenMPControls &operator=(const ScopedOpenMPControls &) = delete;

        ~ScopedOpenMPControls()
        {
            omp_set_num_threads(original_threads_);
            omp_set_dynamic(original_dynamic_);
        }

    private:
        int original_threads_;
        int original_dynamic_;
    };
} // namespace avx2_parity_helpers

using namespace avx2_parity_helpers;

// ============================================================================
// Test fixture
// ============================================================================

class AVX2VNNIParity : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!cpu_supports_avx512_vnni())
        {
            GTEST_SKIP() << "AVX512-VNNI not available; cannot run parity tests";
        }
    }
};

/**
 * @test Prove nominal M=1 candidates collapse only when N makes them aliases.
 *
 * This policy-only test launches no kernel.  It protects profiler ownership:
 * NBC8 and NBC16 at a three-chunk shape are the same physical NBC4 launch and
 * must never receive independent counter records.
 */
TEST(CPUNativeVNNIDecodePolicy, NormalizesGeometryDependentAliases)
{
    const auto resolve = [](DecodeSchedulePolicy policy, int n)
    {
        return resolveDecodeSchedulePolicy(
                   policy,
                   DecodeScheduleGeometry{
                       .n = n,
                       .k = 32,
                       .k_tiles = 0,
                       .threads = 28,
                   })
            .effective;
    };
    EXPECT_EQ(
        resolve(DecodeSchedulePolicy::Nbc16, 64),
        DecodeSchedulePolicy::Nbc1);
    EXPECT_EQ(
        resolve(DecodeSchedulePolicy::Nbc8, 160),
        DecodeSchedulePolicy::Nbc4);
    EXPECT_EQ(
        resolve(DecodeSchedulePolicy::Nbc16, 448),
        DecodeSchedulePolicy::Nbc8);
    EXPECT_EQ(
        resolve(DecodeSchedulePolicy::Nbc2, 448),
        DecodeSchedulePolicy::Nbc2);
    EXPECT_EQ(
        resolve(DecodeSchedulePolicy::Nbc16, 2048),
        DecodeSchedulePolicy::Nbc16);
}

/**
 * @test Keep coarse M=1 schedules from suppressing useful core parallelism.
 *
 * The Qwen development geometry below reproduced a five-minute NBC8/NBC16
 * timing cell because those policies exposed only 22 and 11 full-K producer
 * tasks to a 28-core socket. NBC4 exposes 44 tasks and completes the same
 * byte-exact work economically. Small matrices retain the coarse policies,
 * and a two-tile frozen K partition also makes NBC8 sufficiently parallel.
 */
TEST(CPUNativeVNNIDecodePolicy, ResolvesLargeUnderfilledSchedulesToFullTeamGrid)
{
    constexpr int N = 11264;
    constexpr int K = 38912;
    constexpr int threads = 28;

    const DecodeScheduleGeometry full_k{
        .n = N,
        .k = K,
        .k_tiles = 0,
        .threads = threads,
    };
    for (const DecodeSchedulePolicy policy : {
             DecodeSchedulePolicy::Nbc1,
             DecodeSchedulePolicy::Nbc2,
             DecodeSchedulePolicy::Nbc4})
    {
        const DecodeScheduleResolution resolution =
            resolveDecodeSchedulePolicy(policy, full_k);
        EXPECT_TRUE(resolution.isExactPhysicalIdentity());
        EXPECT_EQ(resolution.effective, policy);
        EXPECT_GE(resolution.producer_tasks, resolution.target_tasks);
    }

    for (const DecodeSchedulePolicy policy : {
             DecodeSchedulePolicy::Nbc8,
             DecodeSchedulePolicy::Nbc16})
    {
        const DecodeScheduleResolution resolution =
            resolveDecodeSchedulePolicy(policy, full_k);
        EXPECT_FALSE(resolution.isExactPhysicalIdentity());
        EXPECT_EQ(resolution.effective, DecodeSchedulePolicy::Nbc4);
        EXPECT_EQ(resolution.n_block_chunks, 4);
        EXPECT_EQ(resolution.producer_tasks, 44);
        EXPECT_EQ(resolution.target_tasks, threads);
    }

    const DecodeScheduleResolution small_work = resolveDecodeSchedulePolicy(
        DecodeSchedulePolicy::Nbc16,
        DecodeScheduleGeometry{
            .n = N,
            .k = 2048,
            .k_tiles = 0,
            .threads = threads,
        });
    EXPECT_TRUE(small_work.isExactPhysicalIdentity());
    EXPECT_EQ(small_work.effective, DecodeSchedulePolicy::Nbc16);

    const DecodeScheduleResolution k_parallel = resolveDecodeSchedulePolicy(
        DecodeSchedulePolicy::Nbc8,
        DecodeScheduleGeometry{
            .n = N,
            .k = K,
            .k_tiles = 2,
            .threads = threads,
        });
    EXPECT_TRUE(k_parallel.isExactPhysicalIdentity());
    EXPECT_EQ(k_parallel.producer_tasks, 44);
}

/**
 * @test Resolve every grouped verifier task topology from one typed authority.
 *
 * The fused bundle scheduler, direct launcher, PerfStats contract, and corpus
 * adapter must agree on more than a policy label. This device-free regression
 * fixes the exact physical row tile, N-block width, producer task count, and
 * reduction task count for each distinct grid. It also proves that zero and
 * one both mean full-K while only values greater than one mean independent K
 * partials.
 */
TEST(CPUNativeVNNIVerifierSchedule, ResolvesPhysicalTaskGridIdentity)
{
    EXPECT_FALSE(nativeVNNIUsesKPartitions(0));
    EXPECT_FALSE(nativeVNNIUsesKPartitions(1));
    EXPECT_TRUE(nativeVNNIUsesKPartitions(2));

    const auto resolve_full_k = [](VerifierRowsPolicy policy)
    {
        return resolveVerifierRowsSchedule(
            policy,
            policy,
            VerifierRowsScheduleGeometry{
                .rows = 5,
                .physical_n = 320,
                .policy_n = 320,
                .k_tiles = 0,
                .ambient_n_block_chunks = 4,
                .use_avx512 = true,
            });
    };

    const VerifierRowsScheduleResolution row_chunk = resolve_full_k(
        VerifierRowsPolicy::FullKRowChunkGrid);
    EXPECT_EQ(
        row_chunk.route,
        VerifierRowsExecutionRoute::GroupedFullKRowChunkGrid);
    EXPECT_EQ(row_chunk.task_grid, VerifierRowsTaskGrid::RowNChunk);
    EXPECT_EQ(row_chunk.physical_row_tile, 1);
    EXPECT_EQ(row_chunk.n_block_chunks, 1);
    EXPECT_EQ(row_chunk.producer_tasks, 25);
    EXPECT_EQ(row_chunk.reduction_tasks, 0);

    const VerifierRowsScheduleResolution n_major = resolve_full_k(
        VerifierRowsPolicy::FullKTwoRowNbc2);
    EXPECT_EQ(
        n_major.route,
        VerifierRowsExecutionRoute::GroupedFullKTwoRowNMajor);
    EXPECT_EQ(n_major.task_grid, VerifierRowsTaskGrid::NBlockAllRows);
    EXPECT_EQ(n_major.physical_row_tile, 2);
    EXPECT_EQ(n_major.n_block_chunks, 2);
    EXPECT_EQ(n_major.n_blocks, 3);
    EXPECT_EQ(n_major.producer_tasks, 3);

    const VerifierRowsScheduleResolution pair_grid = resolve_full_k(
        VerifierRowsPolicy::FullKTwoRowPairGridNbc4);
    EXPECT_EQ(
        pair_grid.route,
        VerifierRowsExecutionRoute::GroupedFullKPairGrid);
    EXPECT_EQ(pair_grid.task_grid, VerifierRowsTaskGrid::RowTileNBlock);
    EXPECT_EQ(pair_grid.physical_row_tile, 2);
    EXPECT_EQ(pair_grid.n_block_chunks, 4);
    EXPECT_EQ(pair_grid.n_blocks, 2);
    EXPECT_EQ(pair_grid.producer_tasks, 6);

    const VerifierRowsScheduleResolution wide = resolve_full_k(
        VerifierRowsPolicy::WideRows);
    EXPECT_EQ(
        wide.route,
        VerifierRowsExecutionRoute::GroupedFullKWideRows);
    EXPECT_EQ(wide.physical_row_tile, 4);
    EXPECT_EQ(wide.n_block_chunks, 4);
    EXPECT_EQ(wide.producer_tasks, 4);

    const auto resolve_kpart = [](VerifierRowsPolicy policy)
    {
        return resolveVerifierRowsSchedule(
            policy,
            policy,
            VerifierRowsScheduleGeometry{
                .rows = 5,
                .physical_n = 320,
                .policy_n = 320,
                .k_tiles = 3,
                .ambient_n_block_chunks = 8,
                .use_avx512 = true,
            });
    };
    const VerifierRowsScheduleResolution pairwise_kpart = resolve_kpart(
        VerifierRowsPolicy::Pairwise);
    EXPECT_EQ(
        pairwise_kpart.route,
        VerifierRowsExecutionRoute::GroupedKParallelRowTiles);
    EXPECT_EQ(
        pairwise_kpart.task_grid,
        VerifierRowsTaskGrid::RowTileNChunkKTile);
    EXPECT_EQ(pairwise_kpart.physical_row_tile, 2);
    EXPECT_EQ(pairwise_kpart.n_block_chunks, 1);
    EXPECT_EQ(pairwise_kpart.producer_tasks, 45);
    EXPECT_EQ(pairwise_kpart.reduction_tasks, 25);

    const VerifierRowsScheduleResolution wide_kpart = resolve_kpart(
        VerifierRowsPolicy::WideRows);
    EXPECT_EQ(wide_kpart.physical_row_tile, 4);
    EXPECT_EQ(wide_kpart.n_block_chunks, 1);
    EXPECT_EQ(wide_kpart.producer_tasks, 30);
    EXPECT_EQ(wide_kpart.reduction_tasks, 25);

    EXPECT_THROW(
        resolve_kpart(VerifierRowsPolicy::FullKTwoRowNbc1),
        std::invalid_argument);
}

/**
 * @test Reject aliases and invalid full-K overrides before kernel execution.
 *
 * A forced candidate is evidence for one physical launch. Explicitly naming a
 * wider grid that collapses on small N must therefore fail, while generated
 * `Auto` dispatch may publish the canonical route. The trainer-only override
 * is legal only for full-K Pairwise/WideRows pair grids.
 */
TEST(CPUNativeVNNIVerifierSchedule, RejectsNonPhysicalExplicitIdentity)
{
    const VerifierRowsScheduleGeometry small_n{
        .rows = 4,
        .physical_n = 64,
        .policy_n = 64,
        .k_tiles = 0,
        .ambient_n_block_chunks = 4,
        .use_avx512 = true,
    };
    EXPECT_THROW(
        resolveVerifierRowsSchedule(
            VerifierRowsPolicy::FullKTwoRowPairGridNbc8,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc8,
            small_n),
        std::invalid_argument);

    const VerifierRowsScheduleResolution generated =
        resolveVerifierRowsSchedule(
            VerifierRowsPolicy::Auto,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc8,
            small_n);
    EXPECT_EQ(
        generated.effective,
        VerifierRowsPolicy::FullKTwoRowPairGridNbc1);
    EXPECT_EQ(generated.n_block_chunks, 1);

    VerifierRowsScheduleGeometry override_geometry = small_n;
    override_geometry.physical_n = 320;
    override_geometry.policy_n = 320;
    override_geometry.full_k_n_block_chunks_override = 2;
    const VerifierRowsScheduleResolution overridden =
        resolveVerifierRowsSchedule(
            VerifierRowsPolicy::Pairwise,
            VerifierRowsPolicy::Pairwise,
            override_geometry);
    EXPECT_EQ(overridden.n_block_chunks, 2);
    EXPECT_EQ(overridden.producer_tasks, 6);

    EXPECT_THROW(
        resolveVerifierRowsSchedule(
            VerifierRowsPolicy::FullKTwoRowPairGridNbc2,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc2,
            override_geometry),
        std::invalid_argument);
}

/**
 * @test Prove learned M=1 dispatch is physically total on unseen geometries.
 *
 * The generated tree names nominal candidate families learned from measured
 * shapes.  This sweep crosses every production runtime codebook, both frozen
 * K-partition regimes, positive runtime thread counts, and N boundaries where
 * wider families become physical aliases. It therefore catches both missing
 * topology coverage and the grouped-seal failure where IQ4_XS at N=160 and
 * K=224 selected nominal NBC8 even though its three N chunks have one physical
 * NBC4 owner.
 */
TEST(CPUNativeVNNIDecodePolicy, GeneratedRulesResolvePhysicalAllCodebooks)
{
    using generated::CPUNativeVNNIDecodeBuildISA;
    using generated::CPUNativeVNNIDecodePolicy;
    using generated::CPUNativeVNNIDecodeRuntimeISA;

    constexpr std::array<uint8_t, 18> codebooks{
        0, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 19, 20, 21,
    };
    constexpr std::array<int, 23> output_widths{
        1, 16, 63, 64, 65, 127, 128, 129, 160, 191, 192, 193,
        255, 256, 257, 511, 512, 513, 816, 880, 2032, 4064, 5120,
    };
    constexpr std::array<int, 9> input_widths{
        32, 224, 256, 800, 992, 2048, 8192, 11216, 16384,
    };
    constexpr std::array<int, 10> thread_counts{
        1, 2, 3, 7, 27, 28, 31, 56, 112, 255,
    };

#if LLAMINAR_COMPILED_WITH_AVX512
    constexpr std::array runtime_regimes{
        std::pair{CPUNativeVNNIDecodeBuildISA::AVX512,
                  CPUNativeVNNIDecodeRuntimeISA::AVX2},
        std::pair{CPUNativeVNNIDecodeBuildISA::AVX512,
                  CPUNativeVNNIDecodeRuntimeISA::AVX512},
    };
#else
    constexpr std::array runtime_regimes{
        std::pair{CPUNativeVNNIDecodeBuildISA::AVX2,
                  CPUNativeVNNIDecodeRuntimeISA::AVX2},
    };
#endif

    for (const auto [build_isa, runtime_isa] : runtime_regimes)
    {
        for (const uint8_t codebook : codebooks)
        {
            for (const int n : output_widths)
            {
                for (const int k : input_widths)
                {
                    for (const int threads : thread_counts)
                    {
                        for (const bool serial_kpart : {false, true})
                        {
                            CPUNativeVNNIDecodePolicy nominal{};
                            ASSERT_TRUE(
                                generated::selectCPUNativeVNNIDecodeGeneratedPolicy(
                                    build_isa,
                                    runtime_isa,
                                    threads,
                                    codebook,
                                    n,
                                    k,
                                    serial_kpart,
                                    serial_kpart ? 2 : 1,
                                    nominal))
                                << "threads=" << threads
                                << " codebook=" << static_cast<int>(codebook)
                                << " N=" << n << " K=" << k
                                << " serial_kpart=" << serial_kpart;

                            const DecodeScheduleGeometry geometry{
                                .n = n,
                                .k = k,
                                .k_tiles = serial_kpart ? 2 : 0,
                                .threads = threads,
                            };
                            const DecodeSchedulePolicy physical =
                                resolveGeneratedDecodeSchedulePolicy(
                                    nominal, geometry);
                            EXPECT_TRUE(
                                resolveDecodeSchedulePolicy(physical, geometry)
                                    .isExactPhysicalIdentity())
                                << "threads=" << threads
                                << " codebook=" << static_cast<int>(codebook)
                                << " N=" << n << " K=" << k
                                << " serial_kpart=" << serial_kpart;
                        }
                    }
                }
            }
        }
    }

    /*
     * Keep the original three-chunk regression independent of the currently
     * installed fit.  A future corpus may legitimately select NBC1 or NBC2 for
     * this exact geometry; the architectural invariant is that a nominal NBC8
     * candidate is resolved to its one physical NBC4 launch before execution.
     */
    EXPECT_EQ(
        resolveGeneratedDecodeSchedulePolicy(
            CPUNativeVNNIDecodePolicy::Nbc8,
            DecodeScheduleGeometry{
                .n = 160,
                .k = 224,
                .k_tiles = 0,
                .threads = 28,
            }),
        DecodeSchedulePolicy::Nbc4);
}

/**
 * @test Prove grouped verifier dispatch is total over its runtime key space.
 *
 * The grouped selector maps every M above 16 to the trained M31 domain. This
 * sweep combines that M-totality rule with every CPU codebook, all four aspect
 * regimes, full-K and K-part geometries, and representative positive OpenMP
 * widths. No exact overlay is required for an unseen topology.
 */
TEST(CPUNativeVNNIVerifierPolicy, GeneratedRulesCoverAllPositiveThreadCounts)
{
    using generated::CPUNativeVNNIBuildISA;
    using generated::CPUNativeVNNIRuntimeISA;
    using generated::CPUNativeVNNIVerifierRowsPolicy;

    constexpr std::array<uint8_t, 18> codebooks{
        0, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 19, 20, 21,
    };
    constexpr std::array<int, 10> thread_counts{
        1, 2, 3, 7, 27, 28, 31, 56, 112, 255,
    };
    constexpr std::array<int, 13> m_values{
        2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 255, 256, 1024,
    };
    constexpr std::array<std::pair<int, int>, 6> geometries{{
        {16, 1024},
        {256, 4096},
        {1024, 1024},
        {4096, 1024},
        {16384, 256},
        {248320, 7168},
    }};

#if LLAMINAR_COMPILED_WITH_AVX512
    constexpr std::array runtime_regimes{
        std::pair{CPUNativeVNNIBuildISA::AVX512,
                  CPUNativeVNNIRuntimeISA::AVX2},
        std::pair{CPUNativeVNNIBuildISA::AVX512,
                  CPUNativeVNNIRuntimeISA::AVX512},
    };
#else
    constexpr std::array runtime_regimes{
        std::pair{CPUNativeVNNIBuildISA::AVX2,
                  CPUNativeVNNIRuntimeISA::AVX2},
    };
#endif

    for (const auto [build_isa, runtime_isa] : runtime_regimes)
    {
        for (const uint8_t codebook : codebooks)
        {
            for (const int M : m_values)
            {
                for (const auto [N, K] : geometries)
                {
                    for (const int threads : thread_counts)
                    {
                        for (const int k_tiles : {0, 2})
                        {
                            CPUNativeVNNIVerifierRowsPolicy policy{};
                            ASSERT_TRUE(
                                generated::
                                    selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                                        build_isa,
                                        runtime_isa,
                                        threads,
                                        codebook,
                                        M,
                                        N,
                                        K,
                                        k_tiles,
                                        policy))
                                << "threads=" << threads << " M=" << M
                                << " codebook=" << static_cast<int>(codebook)
                                << " N=" << N << " K=" << K
                                << " k_tiles=" << k_tiles;
                        }
                    }
                }
            }
        }
    }

    CPUNativeVNNIVerifierRowsPolicy policy{};
    EXPECT_FALSE(
        generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
            runtime_regimes.front().first,
            runtime_regimes.front().second,
            0,
            0,
            2,
            1024,
            1024,
            0,
            policy));
}

/**
 * @test Prove generated dispatch has no practical positive-thread holes.
 *
 * The broad policy tests above cross formats, geometries, M values, and
 * arithmetic regimes at representative thread counts.  This orthogonal sweep
 * holds representative full-K and K-partition geometries fixed while checking
 * every positive thread count through 4096 plus `INT_MAX`.  Together the tests
 * catch both topology-specific rule gaps and integer-boundary failures without
 * multiplying the complete geometry matrix by thousands of redundant widths.
 */
TEST(CPUNativeVNNIThreadTotality, PolicyCacheRetainsFullRuntimeRowIdentity)
{
    constexpr std::array<uint8_t, 18> codebooks{
        0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21};
#if LLAMINAR_COMPILED_WITH_AVX512
    constexpr auto build = generated::CPUNativeVNNIBuildISA::AVX512;
    constexpr std::array isas{ISALevel::AVX2, ISALevel::AVX512};
#else
    constexpr auto build = generated::CPUNativeVNNIBuildISA::AVX2;
    constexpr std::array isas{ISALevel::AVX2};
#endif
    // Routed prefill can exceed the exact-overlay key's eight-bit row field.
    // Alternate aliases in both directions and compare to the uncached oracle.
    constexpr std::array rows{258, 2, 514, 2, 259, 3, 287, 31, 511, 255};
    for (auto isa : isas)
        for (auto codebook : codebooks)
            for (int threads : {1, 28})
                for (int k_tiles : {0, 4})
                    for (int m : rows)
                    {
                        CPUNativeVNNIPackedWeights packed;
                        packed.codebook_id = codebook;
                        const auto runtime = isa == ISALevel::AVX512
                            ? generated::CPUNativeVNNIRuntimeISA::AVX512
                            : generated::CPUNativeVNNIRuntimeISA::AVX2;
                        generated::CPUNativeVNNIVerifierRowsPolicy expected{};
                        ASSERT_TRUE(generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                            build, runtime, threads, codebook, m, 512, 256, k_tiles, expected));
                        const auto actual = selectVerifierRowsPolicy(
                            packed, m, 512, 256, isa, threads, k_tiles);
                        ASSERT_EQ(actual, verifierRowsPolicyFromGenerated(expected))
                            << "codebook=" << int(codebook) << " rows=" << m
                            << " threads=" << threads << " k_tiles=" << k_tiles;
                    }
}

/** @test Generated dispatch is total over positive CPU worker counts. */
TEST(CPUNativeVNNIThreadTotality, GeneratedSelectorsHaveNoPositiveThreadHoles)
{
    using generated::CPUNativeVNNIBuildISA;
    using generated::CPUNativeVNNIDecodeBuildISA;
    using generated::CPUNativeVNNIDecodePolicy;
    using generated::CPUNativeVNNIDecodeRuntimeISA;
    using generated::CPUNativeVNNIRuntimeISA;
    using generated::CPUNativeVNNIVerifierRowsPolicy;

    constexpr std::array<uint8_t, 18> codebooks{
        0, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 19, 20, 21,
    };

#if LLAMINAR_COMPILED_WITH_AVX512
    constexpr std::array decode_runtime_regimes{
        std::pair{CPUNativeVNNIDecodeBuildISA::AVX512,
                  CPUNativeVNNIDecodeRuntimeISA::AVX2},
        std::pair{CPUNativeVNNIDecodeBuildISA::AVX512,
                  CPUNativeVNNIDecodeRuntimeISA::AVX512},
    };
    constexpr std::array verifier_runtime_regimes{
        std::pair{CPUNativeVNNIBuildISA::AVX512,
                  CPUNativeVNNIRuntimeISA::AVX2},
        std::pair{CPUNativeVNNIBuildISA::AVX512,
                  CPUNativeVNNIRuntimeISA::AVX512},
    };
#else
    constexpr std::array decode_runtime_regimes{
        std::pair{CPUNativeVNNIDecodeBuildISA::AVX2,
                  CPUNativeVNNIDecodeRuntimeISA::AVX2},
    };
    constexpr std::array verifier_runtime_regimes{
        std::pair{CPUNativeVNNIBuildISA::AVX2,
                  CPUNativeVNNIRuntimeISA::AVX2},
    };
#endif

    const auto verify_thread_count = [&](int threads)
    {
        for (const uint8_t codebook : codebooks)
        {
            for (const auto [build_isa, runtime_isa] : decode_runtime_regimes)
            {
                CPUNativeVNNIDecodePolicy full_k_policy{};
                ASSERT_TRUE(
                    generated::selectCPUNativeVNNIDecodeGeneratedPolicy(
                        build_isa,
                        runtime_isa,
                        threads,
                        codebook,
                        512,
                        256,
                        false,
                        1,
                        full_k_policy))
                    << "full-K decode threads=" << threads
                    << " codebook=" << static_cast<int>(codebook);

                CPUNativeVNNIDecodePolicy kpart_policy{};
                ASSERT_TRUE(
                    generated::selectCPUNativeVNNIDecodeGeneratedPolicy(
                        build_isa,
                        runtime_isa,
                        threads,
                        codebook,
                        64,
                        8192,
                        true,
                        8,
                        kpart_policy))
                    << "K-part decode threads=" << threads
                    << " codebook=" << static_cast<int>(codebook);
            }

            for (const auto [build_isa, runtime_isa] : verifier_runtime_regimes)
            {
                CPUNativeVNNIVerifierRowsPolicy full_k_policy{};
                ASSERT_TRUE(
                    generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                        build_isa,
                        runtime_isa,
                        threads,
                        codebook,
                        3,
                        512,
                        256,
                        0,
                        full_k_policy))
                    << "full-K verifier threads=" << threads
                    << " codebook=" << static_cast<int>(codebook);

                CPUNativeVNNIVerifierRowsPolicy kpart_policy{};
                ASSERT_TRUE(
                    generated::selectCPUNativeVNNIVerifierRowsGeneratedPolicy(
                        build_isa,
                        runtime_isa,
                        threads,
                        codebook,
                        15,
                        64,
                        8192,
                        8,
                        kpart_policy))
                    << "K-part verifier threads=" << threads
                    << " codebook=" << static_cast<int>(codebook);
            }
        }
    };

    for (int threads = 1; threads <= 4096; ++threads)
        verify_thread_count(threads);
    verify_thread_count(std::numeric_limits<int>::max());
}

/**
 * @test Execute production decode and grouped verifier across CPU team sizes.
 *
 * The generated selector test proves dispatch lookup totality without
 * launching thousands of oversized OpenMP teams.  This execution regression
 * complements it with every thread width from one through the OpenMP runtime's
 * available-processor count, capped at 64 for unit-test latency, plus awkward
 * non-power-of-two widths. Both full-K and serial K-partition geometries run
 * through production `Auto` dispatch. Every grouped row must be byte-identical
 * to a production M=1 invocation under the same runtime thread topology.
 */
TEST(CPUNativeVNNIThreadTotality, ProductionDecodeAndGroupedVerifierExecute)
{
    if (!cpu_supports_avx2())
        GTEST_SKIP() << "NativeVNNI production kernels require AVX2";

    ScopedOpenMPControls openmp_guard;
    constexpr int M = 3;
    const int bounded_runtime_width =
        std::clamp(omp_get_num_procs(), 1, 64);

    std::vector<int> execution_widths;
    execution_widths.reserve(static_cast<size_t>(bounded_runtime_width) + 4);
    for (int threads = 1; threads <= bounded_runtime_width; ++threads)
        execution_widths.push_back(threads);
    for (const int threads : {27, 31, 56, 63})
    {
        if (std::find(execution_widths.begin(), execution_widths.end(), threads) ==
            execution_widths.end())
        {
            execution_widths.push_back(threads);
        }
    }

    struct Geometry
    {
        int n;
        int k;
        uint32_t seed;
        const char *label;
    };
    constexpr std::array geometries{
        Geometry{512, 256, 0x7100u, "full-K"},
        Geometry{64, 8192, 0x7200u, "K-partitioned"},
    };

    for (const Geometry &geometry : geometries)
    {
        SCOPED_TRACE(geometry.label);
        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(geometry.n),
             static_cast<size_t>(geometry.k)},
            geometry.seed);
        ASSERT_NE(weights, nullptr);
        const CPUNativeVNNIPackedWeights packed = packWeights(weights.get());

        std::vector<Q8_1Block> activations(
            static_cast<size_t>(M) * packed.blocks_per_row);
        for (int row = 0; row < M; ++row)
        {
            const auto row_blocks =
                createRandomQ8_1(geometry.k, geometry.seed + row + 1);
            std::copy(
                row_blocks.begin(),
                row_blocks.end(),
                activations.begin() +
                    static_cast<size_t>(row) * packed.blocks_per_row);
        }

        std::vector<float> serial(
            static_cast<size_t>(M) * geometry.n);
        std::vector<float> grouped(
            static_cast<size_t>(M) * geometry.n);

        for (const int threads : execution_widths)
        {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads);
            omp_set_num_threads(threads);
            std::fill(serial.begin(), serial.end(), 0.0f);
            std::fill(grouped.begin(), grouped.end(), 0.0f);

            for (int row = 0; row < M; ++row)
            {
                gemv_native_vnni_preq(
                    packed,
                    activations.data() +
                        static_cast<size_t>(row) * packed.blocks_per_row,
                    serial.data() + static_cast<size_t>(row) * geometry.n,
                    ISAPath::AUTO,
                    DecodeSchedulePolicy::Auto);
            }
            gemm_native_vnni_preq_decode_equivalent_rows(
                packed,
                activations.data(),
                grouped.data(),
                M,
                geometry.n);

            EXPECT_EQ(
                std::memcmp(
                    grouped.data(),
                    serial.data(),
                    grouped.size() * sizeof(float)),
                0)
                << geometry.label << " grouped verifier differs from "
                << "production serial decode at threads=" << threads;
        }
    }
}

/**
 * @test Scalar M=1 execution remains an explicit diagnostic oracle only.
 *
 * Production `Auto` and concrete learned schedule requests must never turn an
 * unsupported runtime ISA into scalar row replay. The storage can stay empty,
 * but the packed geometry must remain valid so this test reaches the scalar
 * policy guard instead of the independent malformed-geometry guard.
 */
TEST(CPUNativeVNNIDecodePolicy, RejectsScalarProductionSchedules)
{
    CPUNativeVNNIPackedWeights packed;
    packed.N = 64;
    packed.K = 32;
    packed.blocks_per_row = 1;
    packed.payload_bytes = 18;
    std::array<Q8_1Block, 1> activation{};
    std::array<float, 64> output{};

    EXPECT_THROW(
        gemv_native_vnni_preq(
            packed,
            activation.data(),
            output.data(),
            ISAPath::SCALAR,
            DecodeSchedulePolicy::Auto),
        std::runtime_error);
    EXPECT_THROW(
        gemv_native_vnni_preq(
            packed,
            activation.data(),
            output.data(),
            ISAPath::SCALAR,
            DecodeSchedulePolicy::Nbc1),
        std::runtime_error);
}

// ============================================================================
// Level 1: Intrinsic-level dpbusd parity
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

TEST_F(AVX2VNNIParity, DpbusdIntrinsic_ZeroAccumulator)
{
    // Test with zero accumulator — pure dot product
    alignas(64) uint8_t a_data[64];
    alignas(64) int8_t b_data[64];
    std::mt19937 rng(123);

    for (int i = 0; i < 64; ++i)
    {
        a_data[i] = static_cast<uint8_t>(rng() % 256);
        b_data[i] = static_cast<int8_t>((rng() % 256) - 128);
    }

    // AVX512: process full 64 bytes as one ZMM
    __m512i acc512 = _mm512_setzero_si512();
    __m512i a512 = _mm512_loadu_si512(a_data);
    __m512i b512 = _mm512_loadu_si512(b_data);
    acc512 = _mm512_dpbusd_epi32(acc512, a512, b512);

    alignas(64) int32_t result_512[16];
    _mm512_store_si512(result_512, acc512);

    // AVX2: process as two 32-byte YMM halves
    __m256i acc256_lo = _mm256_setzero_si256();
    __m256i acc256_hi = _mm256_setzero_si256();
    __m256i a256_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data));
    __m256i a256_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data + 32));
    __m256i b256_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data));
    __m256i b256_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data + 32));

    acc256_lo = avx2_dpbusd_epi32(acc256_lo, a256_lo, b256_lo);
    acc256_hi = avx2_dpbusd_epi32(acc256_hi, a256_hi, b256_hi);

    alignas(32) int32_t result_256[16];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256), acc256_lo);
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256 + 8), acc256_hi);

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(result_512[i], result_256[i])
            << "Lane " << i << ": AVX512=" << result_512[i]
            << " AVX2=" << result_256[i];
    }
}

/**
 * @test Exhaust every scalar input value at the AVX2 emulation boundary.
 *
 * A direct unsigned-byte by signed-byte `maddubs` implementation saturates for
 * combinations such as 255 x 127. The production emulation separates even and
 * odd byte products before INT32 accumulation, so all 65,536 scalar value pairs
 * must still reproduce the exact four-product dot in every vector lane.
 */
TEST_F(AVX2VNNIParity, DpbusdIntrinsic_ExhaustiveUnsignedSignedByteDomain)
{
    alignas(32) std::array<uint8_t, 32> activations{};
    alignas(32) std::array<int8_t, 32> weights{};
    alignas(32) std::array<int32_t, 8> actual{};

    for (int activation = 0; activation <= 255; ++activation)
    {
        activations.fill(static_cast<uint8_t>(activation));
        for (int weight = -128; weight <= 127; ++weight)
        {
            weights.fill(static_cast<int8_t>(weight));
            const __m256i result = avx2_dpbusd_epi32(
                _mm256_setzero_si256(),
                _mm256_load_si256(
                    reinterpret_cast<const __m256i *>(activations.data())),
                _mm256_load_si256(
                    reinterpret_cast<const __m256i *>(weights.data())));
            _mm256_store_si256(
                reinterpret_cast<__m256i *>(actual.data()), result);

            const int32_t expected = 4 * activation * weight;
            for (size_t lane = 0; lane < actual.size(); ++lane)
            {
                if (actual[lane] != expected)
                {
                    FAIL() << "activation=" << activation
                           << " weight=" << weight
                           << " lane=" << lane
                           << " expected=" << expected
                           << " actual=" << actual[lane];
                }
            }
        }
    }
}

TEST_F(AVX2VNNIParity, DpbusdIntrinsic_WithAccumulator)
{
    // Test with non-zero accumulator
    alignas(64) uint8_t a_data[64];
    alignas(64) int8_t b_data[64];
    alignas(64) int32_t acc_init[16];
    std::mt19937 rng(456);

    for (int i = 0; i < 64; ++i)
    {
        a_data[i] = static_cast<uint8_t>(rng() % 256);
        b_data[i] = static_cast<int8_t>((rng() % 256) - 128);
    }
    for (int i = 0; i < 16; ++i)
    {
        acc_init[i] = static_cast<int32_t>(rng() % 100000) - 50000;
    }

    // AVX512
    __m512i acc512 = _mm512_load_si512(acc_init);
    acc512 = _mm512_dpbusd_epi32(acc512,
                                  _mm512_loadu_si512(a_data),
                                  _mm512_loadu_si512(b_data));
    alignas(64) int32_t result_512[16];
    _mm512_store_si512(result_512, acc512);

    // AVX2
    __m256i acc_lo = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc_init));
    __m256i acc_hi = _mm256_load_si256(reinterpret_cast<const __m256i *>(acc_init + 8));
    acc_lo = avx2_dpbusd_epi32(acc_lo,
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data)),
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data)));
    acc_hi = avx2_dpbusd_epi32(acc_hi,
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data + 32)),
                                _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data + 32)));
    alignas(32) int32_t result_256[16];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256 + 8), acc_hi);

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(result_512[i], result_256[i])
            << "Lane " << i;
    }
}

TEST_F(AVX2VNNIParity, DpbusdIntrinsic_MultipleAccumulations)
{
    // Simulate K-block accumulation: multiple dpbusd calls on the same accumulator
    constexpr int K_ITERS = 16;
    alignas(64) uint8_t a_data[K_ITERS][64];
    alignas(64) int8_t b_data[K_ITERS][64];
    std::mt19937 rng(789);

    for (int k = 0; k < K_ITERS; ++k)
    {
        for (int i = 0; i < 64; ++i)
        {
            a_data[k][i] = static_cast<uint8_t>(rng() % 256);
            b_data[k][i] = static_cast<int8_t>((rng() % 256) - 128);
        }
    }

    // AVX512
    __m512i acc512 = _mm512_setzero_si512();
    for (int k = 0; k < K_ITERS; ++k)
    {
        acc512 = _mm512_dpbusd_epi32(acc512,
                                      _mm512_loadu_si512(a_data[k]),
                                      _mm512_loadu_si512(b_data[k]));
    }
    alignas(64) int32_t result_512[16];
    _mm512_store_si512(result_512, acc512);

    // AVX2
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for (int k = 0; k < K_ITERS; ++k)
    {
        acc_lo = avx2_dpbusd_epi32(acc_lo,
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data[k])),
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data[k])));
        acc_hi = avx2_dpbusd_epi32(acc_hi,
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(a_data[k] + 32)),
                                    _mm256_load_si256(reinterpret_cast<const __m256i *>(b_data[k] + 32)));
    }
    alignas(32) int32_t result_256[16];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i *>(result_256 + 8), acc_hi);

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_EQ(result_512[i], result_256[i])
            << "Lane " << i << " after " << K_ITERS << " accumulations";
    }
}

// ============================================================================
// Level 2: Single-chunk GEMV parity (64 columns)
// ============================================================================

// Macro to generate chunk-level parity tests for each quant format
#define CHUNK_PARITY_TEST(FORMAT, CREATE_FN, N, K)                                       \
    TEST_F(AVX2VNNIParity, ChunkGEMV_##FORMAT##_##N##x##K)                               \
    {                                                                                      \
        auto weights = TestTensorFactory::CREATE_FN({N, K});                               \
        auto packed = packWeights(weights.get());                                          \
        auto A_q8 = createRandomQ8_1(K, 100);                                             \
                                                                                           \
        alignas(64) float result_512[64] = {};                                             \
        alignas(64) float result_256[64] = {};                                             \
                                                                                           \
        /* AVX512 chunk */                                                                 \
        __m512i lut512 = packed.usesNibbleLUT()                                              \
                             ? build_decode_lut(packed.codebook_id)                        \
                             : _mm512_setzero_si512();                                     \
        if (packed.usesNibbleLUT())                                                          \
            gemv_native_vnni_avx512_chunk_native(packed, A_q8.data(), result_512,          \
                                                 0, 0, packed.blocks_per_row, lut512);     \
        else                                                                               \
            gemv_native_vnni_avx512_chunk_non_nibble(                                    \
                packed, A_q8.data(), result_512,                                          \
                0, 0, packed.blocks_per_row);                                             \
                                                                                           \
        /* AVX2 chunk */                                                                   \
        __m256i lut256 = packed.usesNibbleLUT()                                              \
                             ? build_decode_lut_avx2_for_codebook(packed.codebook_id)      \
                             : _mm256_setzero_si256();                                     \
        if (packed.usesNibbleLUT())                                                          \
            gemv_avx2_chunk_native(packed, A_q8.data(), result_256,                        \
                                   0, 0, packed.blocks_per_row, lut256);                   \
        else                                                                               \
            gemv_avx2_chunk_non_nibble(                                                   \
                packed, A_q8.data(), result_256,                                          \
                0, 0, packed.blocks_per_row);                                             \
                                                                                           \
        assertExactEqual(result_512, result_256, 64,                                       \
                         #FORMAT " chunk GEMV " #N "x" #K);                                \
    }

// Nibble-LUT formats
CHUNK_PARITY_TEST(Q4_0, createQ4_0Random, 64, 256)
CHUNK_PARITY_TEST(Q4_0, createQ4_0Random, 64, 512)
CHUNK_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 64, 256)
CHUNK_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 64, 512)

// Additional nibble-LUT formats
CHUNK_PARITY_TEST(Q4_1, createQ4_1Random, 64, 256)
CHUNK_PARITY_TEST(IQ4_XS, createIQ4_XSRandom, 64, 256)

// INT8 pre-decoded formats (per-block)
CHUNK_PARITY_TEST(Q5_0, createQ5_0Random, 64, 256)
CHUNK_PARITY_TEST(Q5_0, createQ5_0Random, 64, 512)
CHUNK_PARITY_TEST(Q5_1, createQ5_1Random, 64, 256)

// K-quant formats (INT8 pre-decoded, 256-element superblocks)
CHUNK_PARITY_TEST(Q6_K, createQ6_KRandom, 64, 256)
CHUNK_PARITY_TEST(Q5_K, createQ5_KRandom, 64, 256)
CHUNK_PARITY_TEST(Q4_K, createQ4_KRandom, 64, 256)
CHUNK_PARITY_TEST(Q3_K, createQ3_KRandom, 64, 256)
CHUNK_PARITY_TEST(Q2_K, createQ2_KRandom, 64, 256)

// IQ formats (INT8 pre-decoded, 256-element superblocks)
CHUNK_PARITY_TEST(IQ3_S, createIQ3_SRandom, 64, 256)
CHUNK_PARITY_TEST(IQ3_XXS, createIQ3_XXSRandom, 64, 256)
CHUNK_PARITY_TEST(IQ2_S, createIQ2_SRandom, 64, 256)
CHUNK_PARITY_TEST(IQ2_XS, createIQ2_XSRandom, 64, 256)
CHUNK_PARITY_TEST(IQ2_XXS, createIQ2_XXSRandom, 64, 256)
CHUNK_PARITY_TEST(IQ1_S, createIQ1_SRandom, 64, 256)
CHUNK_PARITY_TEST(IQ1_M, createIQ1_MRandom, 64, 256)

#undef CHUNK_PARITY_TEST

// ============================================================================
// Level 3: Full GEMV (M=1) parity via ISAPath dispatch
// ============================================================================

#define FULL_GEMV_PARITY_TEST(FORMAT, CREATE_FN, N, K, SEED)                              \
    TEST_F(AVX2VNNIParity, FullGEMV_##FORMAT##_##N##x##K)                                 \
    {                                                                                      \
        auto weights = TestTensorFactory::CREATE_FN({N, K}, SEED);                         \
        auto packed = packWeights(weights.get());                                          \
        auto A_q8 = createRandomQ8_1(K, SEED + 1);                                        \
                                                                                           \
        std::vector<float> result_512(N, 0.0f);                                            \
        std::vector<float> result_256(N, 0.0f);                                            \
                                                                                           \
        gemv_native_vnni_preq(packed, A_q8.data(), result_512.data(),                      \
                              ISAPath::AVX512,                                             \
                              DecodeSchedulePolicy::FrozenSerialOracle);                  \
        gemv_native_vnni_preq(packed, A_q8.data(), result_256.data(),                      \
                              ISAPath::AVX2,                                               \
                              DecodeSchedulePolicy::FrozenSerialOracle);                  \
                                                                                           \
        assertExactEqual(result_512.data(), result_256.data(), N,                           \
                         #FORMAT " full GEMV " #N "x" #K);                                 \
    }

// Small N (single chunk, no tiling needed)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 64, 256, 42)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 64, 256, 43)

// Medium N (multiple chunks, exercises N-parallel path)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 512, 512, 44)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 512, 512, 45)
FULL_GEMV_PARITY_TEST(Q5_0, createQ5_0Random, 512, 512, 46)
FULL_GEMV_PARITY_TEST(Q5_1, createQ5_1Random, 512, 512, 47)

// Large N (exercises tiling, may trigger K-parallel path)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 4096, 896, 48)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 4096, 896, 49)
FULL_GEMV_PARITY_TEST(Q5_0, createQ5_0Random, 4096, 896, 50)

// Non-64-aligned N (exercises partial chunk handling)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 100, 256, 51)
FULL_GEMV_PARITY_TEST(Q4_0, createQ4_0Random, 200, 512, 52)
FULL_GEMV_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 200, 512, 53)

// Additional nibble-LUT formats
FULL_GEMV_PARITY_TEST(Q4_1, createQ4_1Random, 512, 512, 80)
FULL_GEMV_PARITY_TEST(IQ4_XS, createIQ4_XSRandom, 512, 512, 81)

// K-quant formats (256-element superblocks, INT8 pre-decoded)
FULL_GEMV_PARITY_TEST(Q6_K, createQ6_KRandom, 512, 512, 82)
FULL_GEMV_PARITY_TEST(Q5_K, createQ5_KRandom, 512, 512, 83)
FULL_GEMV_PARITY_TEST(Q4_K, createQ4_KRandom, 512, 512, 84)
FULL_GEMV_PARITY_TEST(Q3_K, createQ3_KRandom, 512, 512, 85)
FULL_GEMV_PARITY_TEST(Q2_K, createQ2_KRandom, 512, 512, 86)

// IQ formats (256-element superblocks, INT8 pre-decoded)
FULL_GEMV_PARITY_TEST(IQ3_S, createIQ3_SRandom, 512, 512, 87)
FULL_GEMV_PARITY_TEST(IQ3_XXS, createIQ3_XXSRandom, 512, 512, 88)
FULL_GEMV_PARITY_TEST(IQ2_S, createIQ2_SRandom, 512, 512, 89)
FULL_GEMV_PARITY_TEST(IQ2_XS, createIQ2_XSRandom, 512, 512, 90)
FULL_GEMV_PARITY_TEST(IQ2_XXS, createIQ2_XXSRandom, 512, 512, 91)
FULL_GEMV_PARITY_TEST(IQ1_S, createIQ1_SRandom, 512, 512, 92)
FULL_GEMV_PARITY_TEST(IQ1_M, createIQ1_MRandom, 512, 512, 93)

// Large K-quant and IQ tests (non-aligned N, larger dimensions)
FULL_GEMV_PARITY_TEST(Q6_K, createQ6_KRandom, 4096, 1024, 94)
FULL_GEMV_PARITY_TEST(Q3_K, createQ3_KRandom, 200, 512, 95)
FULL_GEMV_PARITY_TEST(IQ3_S, createIQ3_SRandom, 200, 512, 96)

#undef FULL_GEMV_PARITY_TEST

// ============================================================================
// Level 4: Full GEMM (M>1) parity via ISAPath dispatch
// ============================================================================

#define FULL_GEMM_PARITY_TEST(FORMAT, CREATE_FN, M, N, K, SEED)                           \
    TEST_F(AVX2VNNIParity, FullGEMM_##FORMAT##_M##M##_##N##x##K)                          \
    {                                                                                      \
        auto weights = TestTensorFactory::CREATE_FN({N, K}, SEED);                         \
        auto packed = packWeights(weights.get());                                          \
        int K_blocks = packed.blocks_per_row;                                              \
                                                                                           \
        /* Create M rows of Q8_1 activations */                                            \
        std::vector<Q8_1Block> A_q8_all(static_cast<size_t>(M) * K_blocks);                \
        std::mt19937 rng(SEED + 100);                                                      \
        std::uniform_int_distribution<int> dist(-127, 127);                                \
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.5f);                    \
        for (int m = 0; m < M; ++m)                                                        \
        {                                                                                  \
            for (int kb = 0; kb < K_blocks; ++kb)                                          \
            {                                                                              \
                auto &blk = A_q8_all[m * K_blocks + kb];                                   \
                float scale = scale_dist(rng);                                             \
                blk.d = simd::fp32_to_fp16(scale);                                         \
                int32_t sum = 0;                                                           \
                for (int i = 0; i < 32; ++i)                                               \
                {                                                                          \
                    blk.qs[i] = static_cast<int8_t>(dist(rng));                            \
                    sum += blk.qs[i];                                                      \
                }                                                                          \
                blk.sum_qs = static_cast<int16_t>(std::clamp(sum, -32768, 32767));         \
            }                                                                              \
        }                                                                                  \
                                                                                           \
        int ldc = N;                                                                       \
        std::vector<float> result_512(static_cast<size_t>(M) * N, 0.0f);                   \
        std::vector<float> result_256(static_cast<size_t>(M) * N, 0.0f);                   \
                                                                                           \
        gemm_native_vnni_preq(packed, A_q8_all.data(), result_512.data(),                  \
                              M, ldc, ISAPath::AVX512);                                    \
        gemm_native_vnni_preq(packed, A_q8_all.data(), result_256.data(),                  \
                              M, ldc, ISAPath::AVX2);                                      \
                                                                                           \
        assertExactEqual(result_512.data(), result_256.data(),                              \
                         static_cast<int>(M) * N,                                          \
                         #FORMAT " GEMM M=" #M " " #N "x" #K);                            \
    }

// M=2 (exercises 2-row microkernel)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 2, 512, 512, 60)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 2, 512, 512, 61)
FULL_GEMM_PARITY_TEST(Q5_0, createQ5_0Random, 2, 512, 512, 62)
FULL_GEMM_PARITY_TEST(Q5_1, createQ5_1Random, 2, 512, 512, 63)

// M=3 (exercises 2-row + 1-row tail)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 3, 512, 512, 64)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 3, 512, 512, 65)

// M=8 (exercises multiple 2-row pairs)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 8, 512, 512, 66)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 8, 512, 512, 67)
FULL_GEMM_PARITY_TEST(Q5_0, createQ5_0Random, 8, 512, 512, 68)

// Large GEMM (realistic LLM dimensions)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 16, 4096, 896, 70)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 16, 4096, 896, 71)

// Non-64-aligned N
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 4, 200, 256, 72)
FULL_GEMM_PARITY_TEST(IQ4_NL, createIQ4_NLRandom, 4, 200, 256, 73)

// M=1 via GEMM path (should match GEMV)
FULL_GEMM_PARITY_TEST(Q4_0, createQ4_0Random, 1, 512, 512, 74)

// Additional nibble-LUT formats
FULL_GEMM_PARITY_TEST(Q4_1, createQ4_1Random, 2, 512, 512, 100)
FULL_GEMM_PARITY_TEST(IQ4_XS, createIQ4_XSRandom, 2, 512, 512, 101)

// K-quant formats (M=2 exercises 2-row microkernel)
FULL_GEMM_PARITY_TEST(Q6_K, createQ6_KRandom, 2, 512, 512, 102)
FULL_GEMM_PARITY_TEST(Q5_K, createQ5_KRandom, 2, 512, 512, 103)
FULL_GEMM_PARITY_TEST(Q4_K, createQ4_KRandom, 2, 512, 512, 104)
FULL_GEMM_PARITY_TEST(Q3_K, createQ3_KRandom, 2, 512, 512, 105)
FULL_GEMM_PARITY_TEST(Q2_K, createQ2_KRandom, 2, 512, 512, 106)

// IQ formats (M=2)
FULL_GEMM_PARITY_TEST(IQ3_S, createIQ3_SRandom, 2, 512, 512, 107)
FULL_GEMM_PARITY_TEST(IQ3_XXS, createIQ3_XXSRandom, 2, 512, 512, 108)
FULL_GEMM_PARITY_TEST(IQ2_S, createIQ2_SRandom, 2, 512, 512, 109)
FULL_GEMM_PARITY_TEST(IQ2_XS, createIQ2_XSRandom, 2, 512, 512, 110)
FULL_GEMM_PARITY_TEST(IQ2_XXS, createIQ2_XXSRandom, 2, 512, 512, 111)
FULL_GEMM_PARITY_TEST(IQ1_S, createIQ1_SRandom, 2, 512, 512, 112)
FULL_GEMM_PARITY_TEST(IQ1_M, createIQ1_MRandom, 2, 512, 512, 113)

// M=3 (2-row + 1-row tail) for representative K-quants and IQ formats
FULL_GEMM_PARITY_TEST(Q6_K, createQ6_KRandom, 3, 512, 512, 114)
FULL_GEMM_PARITY_TEST(Q3_K, createQ3_KRandom, 3, 512, 512, 115)
FULL_GEMM_PARITY_TEST(IQ3_S, createIQ3_SRandom, 3, 512, 512, 116)

// M=8 (multiple 2-row pairs) for representative formats
FULL_GEMM_PARITY_TEST(Q6_K, createQ6_KRandom, 8, 512, 512, 117)
FULL_GEMM_PARITY_TEST(Q4_1, createQ4_1Random, 8, 512, 512, 118)
FULL_GEMM_PARITY_TEST(IQ4_XS, createIQ4_XSRandom, 8, 512, 512, 119)
FULL_GEMM_PARITY_TEST(Q2_K, createQ2_KRandom, 8, 512, 512, 120)
FULL_GEMM_PARITY_TEST(IQ1_S, createIQ1_SRandom, 8, 512, 512, 121)

#undef FULL_GEMM_PARITY_TEST

/**
 * @test Prove Q6_K remains native, dual-scale, and byte exact for every MTP M.
 *
 * This regression deliberately verifies both the physical prepared-weight
 * representation and the arithmetic contract. A Q6_K tensor must not silently
 * regress to the old one-byte-per-weight expansion: each 64-column by 32-K
 * block contains 1,536 bytes of native low/high-bit payload followed by two
 * independent 128-byte FP16 scale arrays. No duplicate scalar payload or INT8
 * expansion is retained.
 *
 * Every grouped runtime size is then evaluated through both explicit ISA
 * dispatches. Each grouped output must match independently executed serial
 * M=1 rows byte for byte, and AVX2 must match AVX-512 byte for byte. This
 * catches changes to reduction order, accidental FMA contraction, row-tail
 * decomposition bugs, and ISA-specific interpretation of the two Q6 scales.
 */
TEST_F(AVX2VNNIParity, Q6KNativeDualScaleEncodingIsByteExactForAllRuntimeRows)
{
    constexpr int N = 256;
    constexpr int K = 512;
    constexpr int native_data_stride = 1536;
    constexpr int native_block_stride = 1792;
    constexpr std::array<int, 17> runtime_rows = {
        2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 31};
    constexpr int maximum_rows = runtime_rows.back();

    ScopedOpenMPControls openmp_controls;
    omp_set_num_threads(4);

    auto weights = TestTensorFactory::createQ6_KRandom({N, K}, 0x5136u);
    const auto packed = packWeights(weights.get());

    ASSERT_TRUE(packed.usesQ6KNativeDualScale());
    EXPECT_FALSE(packed.usesNibbleLUT());
    EXPECT_FALSE(packed.usesExpandedInt8());
    EXPECT_FALSE(packed.usesInlineCompensation());
    EXPECT_EQ(packed.data_stride, native_data_stride);
    EXPECT_EQ(packed.interleaved_block_stride, native_block_stride);
    EXPECT_TRUE(packed.payload.empty());
    EXPECT_TRUE(packed.int8_flat.empty());
    EXPECT_EQ(
        packed.native_interleaved.size(),
        static_cast<size_t>(packed.N_padded / 64) *
            static_cast<size_t>(packed.blocks_per_row) * native_block_stride);

    std::vector<Q8_1Block> activations(
        static_cast<size_t>(maximum_rows) * packed.blocks_per_row);
    for (int row = 0; row < maximum_rows; ++row)
    {
        const auto row_blocks = createRandomQ8_1(K, 0x7000u + row);
        std::copy(
            row_blocks.begin(),
            row_blocks.end(),
            activations.begin() +
                static_cast<size_t>(row) * packed.blocks_per_row);
    }

    std::vector<float> serial_avx512(
        static_cast<size_t>(maximum_rows) * N, 0.0f);
    std::vector<float> serial_avx2(
        static_cast<size_t>(maximum_rows) * N, 0.0f);
    for (int row = 0; row < maximum_rows; ++row)
    {
        const Q8_1Block *const row_activations =
            activations.data() +
            static_cast<size_t>(row) * packed.blocks_per_row;
        gemv_native_vnni_preq(
            packed,
            row_activations,
            serial_avx512.data() + static_cast<size_t>(row) * N,
            ISAPath::AVX512,
            DecodeSchedulePolicy::FrozenSerialOracle);
        gemv_native_vnni_preq(
            packed,
            row_activations,
            serial_avx2.data() + static_cast<size_t>(row) * N,
            ISAPath::AVX2,
            DecodeSchedulePolicy::FrozenSerialOracle);
    }

    ASSERT_EQ(
        std::memcmp(
            serial_avx512.data(),
            serial_avx2.data(),
            serial_avx512.size() * sizeof(float)),
        0)
        << "native Q6_K serial M=1 bytes differ between AVX-512 and AVX2";

    for (const int M : runtime_rows)
    {
        SCOPED_TRACE(std::string("M=") + std::to_string(M));
        const size_t output_elements = static_cast<size_t>(M) * N;
        const size_t output_bytes = output_elements * sizeof(float);
        std::vector<float> grouped_avx512(output_elements, 0.0f);
        std::vector<float> grouped_avx2(output_elements, 0.0f);

        gemm_native_vnni_preq_decode_equivalent_rows(
            packed,
            activations.data(),
            grouped_avx512.data(),
            M,
            N,
            ISAPath::AVX512,
            VerifierRowsPolicy::Auto);
        gemm_native_vnni_preq_decode_equivalent_rows(
            packed,
            activations.data(),
            grouped_avx2.data(),
            M,
            N,
            ISAPath::AVX2,
            VerifierRowsPolicy::Auto);

        ASSERT_EQ(
            std::memcmp(
                grouped_avx512.data(), serial_avx512.data(), output_bytes),
            0)
            << "AVX-512 grouped Q6_K differs from serial M=1 rows";
        ASSERT_EQ(
            std::memcmp(grouped_avx2.data(), serial_avx2.data(), output_bytes),
            0)
            << "AVX2 grouped Q6_K differs from serial M=1 rows";
        ASSERT_EQ(
            std::memcmp(
                grouped_avx512.data(), grouped_avx2.data(), output_bytes),
            0)
            << "grouped Q6_K bytes differ between AVX-512 and AVX2";
    }
}

TEST_F(AVX2VNNIParity, RuntimeISADispatch_AVX2AVX512_ScalarM1OracleOnly)
{
    struct Case
    {
        std::unique_ptr<TensorBase> weights;
        std::string name;
    };

    std::vector<Case> cases;
    cases.push_back({TestTensorFactory::createQ4_0Random({200, 512}, 130), "Q4_0"});
    cases.push_back({TestTensorFactory::createQ6_KRandom({200, 512}, 131), "Q6_K"});

    for (const auto &test_case : cases)
    {
        const auto packed = packWeights(test_case.weights.get());
        const int N = packed.N;
        const int K_blocks = packed.blocks_per_row;
        constexpr int M = 4;

        /*
         * Use one shared activation inventory for every ISA path.  This mirrors
         * MTP verifier publication: rows are already quantized once, then the
         * selected runtime ISA should only change how the packed weights are
         * consumed.
         */
        std::vector<Q8_1Block> A_q8_all(static_cast<size_t>(M) * K_blocks);
        for (int row = 0; row < M; ++row)
        {
            auto row_q8 = createRandomQ8_1(packed.K, 900 + row);
            std::copy(row_q8.begin(), row_q8.end(), A_q8_all.begin() + row * K_blocks);
        }

        std::vector<float> gemv_512(N, 0.0f);
        std::vector<float> gemv_256(N, 0.0f);
        std::vector<float> gemv_scalar(N, 0.0f);
        gemv_native_vnni_preq(
            packed, A_q8_all.data(), gemv_512.data(), ISAPath::AVX512,
            DecodeSchedulePolicy::FrozenSerialOracle);
        gemv_native_vnni_preq(
            packed, A_q8_all.data(), gemv_256.data(), ISAPath::AVX2,
            DecodeSchedulePolicy::FrozenSerialOracle);

        assertExactEqual(gemv_512.data(), gemv_256.data(), N,
                         test_case.name + " M=1 AVX512 vs AVX2");
        gemv_native_vnni_preq(
            packed, A_q8_all.data(), gemv_scalar.data(), ISAPath::SCALAR,
            DecodeSchedulePolicy::FrozenSerialOracle);
        assertStrictMetricClose(gemv_512.data(), gemv_scalar.data(), N,
                                test_case.name + " M=1 AVX512 vs scalar oracle");

        std::vector<float> rows_512(static_cast<size_t>(M) * N, 0.0f);
        std::vector<float> rows_256(static_cast<size_t>(M) * N, 0.0f);
        gemm_native_vnni_preq_decode_equivalent_rows(
            packed, A_q8_all.data(), rows_512.data(), M, N, ISAPath::AVX512);
        gemm_native_vnni_preq_decode_equivalent_rows(
            packed, A_q8_all.data(), rows_256.data(), M, N, ISAPath::AVX2);

        assertExactEqual(rows_512.data(), rows_256.data(), M * N,
                         test_case.name + " verifier rows AVX512 vs AVX2");
        EXPECT_THROW(
            gemm_native_vnni_preq_decode_equivalent_rows(
                packed, A_q8_all.data(), rows_256.data(), M, N,
                ISAPath::SCALAR),
            std::runtime_error)
            << test_case.name
            << " grouped production dispatch must reject the scalar diagnostic path";
    }
}

// ============================================================================
// Level 5: Decode LUT builder parity
// ============================================================================

TEST_F(AVX2VNNIParity, DecodeLUT_Q4_0)
{
    // Verify build_decode_lut_avx2 produces the same per-lane mapping
    alignas(16) static constexpr int8_t expected[16] = {
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

    __m256i lut = build_decode_lut_avx2(expected);
    alignas(32) int8_t result[32];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result), lut);

    // AVX2 vpshufb within 128-bit lanes: both halves should match
    for (int lane = 0; lane < 2; ++lane)
    {
        for (int i = 0; i < 16; ++i)
        {
            EXPECT_EQ(result[lane * 16 + i], expected[i])
                << "Lane " << lane << " index " << i;
        }
    }
}

TEST_F(AVX2VNNIParity, DecodeLUT_IQ4_NL)
{
    alignas(16) static constexpr int8_t expected[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    __m256i lut = build_decode_lut_avx2(expected);
    alignas(32) int8_t result[32];
    _mm256_store_si256(reinterpret_cast<__m256i *>(result), lut);

    for (int lane = 0; lane < 2; ++lane)
    {
        for (int i = 0; i < 16; ++i)
        {
            EXPECT_EQ(result[lane * 16 + i], expected[i])
                << "Lane " << lane << " index " << i;
        }
    }
}

// ============================================================================
// Level 6: Horizontal reduction parity
// ============================================================================

TEST_F(AVX2VNNIParity, HsumPS)
{
    alignas(32) float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    __m256 v = _mm256_load_ps(data);
    float result = hsum_ps_avx2(v);
    float expected = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8;
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(AVX2VNNIParity, HsumEpi32)
{
    alignas(32) int32_t data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(data));
    int32_t result = hsum_epi32_avx2(v);
    int32_t expected = 10 + 20 + 30 + 40 + 50 + 60 + 70 + 80;
    EXPECT_EQ(result, expected);
}

#endif // AVX512F && AVX512VNNI && AVX512BW
