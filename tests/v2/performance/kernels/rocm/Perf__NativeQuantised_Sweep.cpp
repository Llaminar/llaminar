/**
 * @file Perf__NativeQuantised_Sweep.cpp
 * @brief Per-shape tuning sweep for native-quantised GEMM prefill.
 *
 * Benchmarks all combinations of:
 *   - N_TILE: 64 vs 128
 *   - M_TILE: {16, 32, 64}
 *   - MIN_BLOCKS: {1 (bare launch_bounds), 2 (default 2-wave)}
 *   - UNROLL_G: {0 (no hint), 1, 2, 4 (default)} for both launch-bound regimes
 *   - Auto dispatch (current heuristic baseline)
 *
 * Shapes tested: Qwen2.5/Qwen3.6 dense and MoE-style production shapes.
 * M values: MTP small rows plus the canonical graph-prefill bucket policy.
 *
 * Output: per-shape best variant table. Every candidate must be byte-identical
 * to the production exact-M Auto route before its timing can win.
 * Installable timings replay retained native graphs and record their complete
 * projection count, excluding host launch gaps from candidate comparisons.
 * The separate captured-production probe reuses this preparation and byte
 * oracle, but records the installed Auto route for isolated shard economics.
 * Its raw sample CSV is diagnostic evidence, never an installable tournament.
 * Q8 source-default records use their blockwise INT8 producer inventory. An
 * explicit native_quantised preparation scope retains every format's original
 * compact payload/scales and admits native gfx1100 WMMA candidates. These distinct
 * prepared ABIs retain separate plan/CSV identity and byte oracles; neither
 * family can reuse or reinterpret the other family's clocks.
 * Aggregate and event CSVs retain source codebook and physical execution ID
 * separately. IQ4_XS source4 executes24; source-policy4 remains the arithmetic
 * schedule authority. Candidate inspection proves the actual physical launch
 * before either CSV row can be published, never reinterpreting old measurements.
 *
 * Cold policy admission hashes the linked selector table independently of the
 * trainer-local inventory; all timed paths still use the production selector.
 *
 * @note Kernel measurements require a ROCm device. Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#include "kernels/rocm/gemm/ROCmDensePrefillPreparedWeightFamily.h"
#include "kernels/rocm/gemm/ROCmVNNIPrefillLaunch.h"
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlay.h"
#include "kernels/rocm/gemm/ROCmDensePrefillPolicyEvidence.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedWmmaPolicy.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/Sha256.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/ScopedGPUStream.h"
#include "../native_quantised_dispatch/GPUTrainerVerification.h"
#include "../native_quantised_dispatch/NativeQuantisedProfilerControl.h"
#include "../../../utils/DensePrefillTiming.h"
#include "../native_quantised_dispatch/NativeQuantisedShapeManifest.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
    // =========================================================================
    // Constants
    // =========================================================================

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    // =========================================================================
    // Canonical exact-overlay shape inventory
    // =========================================================================

    struct GEMMShape
    {
        std::string name;
        std::string category;
        int N;
        int K;
    };

    /**
     * @brief Resolve every production exact-overlay geometry from the same
     * manifest consumed by CPU, CUDA, and the Python matrix planner.
     *
     * The refresh transaction passes the ordinary-prefill subset explicitly.
     * Keeping the complete production inventory here allows those filters to
     * resolve every released geometry without maintaining a second ROCm-only
     * model table that can drift out of date.
     */
    static const std::vector<GEMMShape> GEMM_SHAPES = []
    {
        std::vector<GEMMShape> result;
        for (const auto &shape :
             llaminar2::test::native_quantised_dispatch::nativeQuantisedShapeManifest())
        {
            if (shape.role != "production" || !shape.exact_overlay)
                continue;
            result.push_back(
                {shape.name, shape.aspect_bucket, shape.N, shape.K});
        }
        if (result.empty())
            throw std::runtime_error(
                "ROCm NativeQuantised manifest has no production exact overlays");
        return result;
    }();

    // Decode and grouped verifier depths have dedicated common-trainer
    // transactions. This legacy harness is restricted to ordinary prefill.
    static const std::vector<int> M_VALUES = supportedPrefillGraphBucketSizes();

    struct FormatSpec
    {
        std::string name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    static const std::vector<FormatSpec> NATIVE_QUANTISED_FORMATS = {
        {"Q4_0", [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"IQ4_XS", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_XSRandom({N, K}); }},
        {"Q4_1", [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q4_K", [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_0", [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},
        {"Q5_K", [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},
        {"IQ3_S", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},
        {"IQ1_S", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
        {"Q8_0", [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_0Random({N, K}); }},
        {"Q8_1", [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_1Random({N, K}); }},
        {"Q8_K", [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_KRandom({N, K}); }},
    };

    static std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    static std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return trim(raw);
    }

    static std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    static std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = toLower(trim(token));
            if (!token.empty())
                values.insert(token);
        }
        return values;
    }

    static std::vector<int> getEnvCsvInts(const char *name, const std::vector<int> &fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return fallback;

        std::vector<int> values;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (!token.empty())
                values.push_back(std::atoi(token.c_str()));
        }
        return values.empty() ? fallback : values;
    }

    static bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    static const NativeQuantisedFormatInfo &requireNativeQuantisedInfo(const TensorBase *weights, const std::string &format_name)
    {
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        const NativeQuantisedFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info)
            throw std::runtime_error("ROCm NativeQuantised sweep format " + format_name + " did not expose vnniFormatInfo()");
        return *info;
    }

    TEST(NativeQuantisedSweepOffline, ShapeAndFormatInventoriesAreCanonical)
    {
        const std::set<std::string> expected_formats = {
            "Q4_0", "IQ4_NL", "IQ4_XS", "Q4_1", "Q4_K", "Q5_0", "Q5_1",
            "Q5_K", "Q6_K", "Q3_K", "Q2_K", "IQ3_S", "IQ3_XXS", "IQ2_S",
            "IQ2_XS", "IQ2_XXS", "IQ1_S", "IQ1_M", "Q8_0", "Q8_1", "Q8_K"};
        std::set<std::string> observed_formats;
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            auto weights = format.create(/*N=*/2, /*K=*/256);
            ASSERT_NE(weights, nullptr) << format.name;
            (void)requireNativeQuantisedInfo(weights.get(), format.name);
            EXPECT_TRUE(observed_formats.insert(format.name).second)
                << "duplicate ROCm NativeQuantised source format " << format.name;
        }
        EXPECT_EQ(observed_formats, expected_formats);

        std::set<std::tuple<std::string, int, int>> expected_shapes;
        for (const auto &shape :
             llaminar2::test::native_quantised_dispatch::nativeQuantisedShapeManifest())
        {
            if (shape.role == "production" && shape.exact_overlay)
                expected_shapes.emplace(shape.name, shape.N, shape.K);
        }
        std::set<std::tuple<std::string, int, int>> observed_shapes;
        for (const auto &shape : GEMM_SHAPES)
            EXPECT_TRUE(
                observed_shapes.emplace(shape.name, shape.N, shape.K).second)
                << "duplicate ROCm NativeQuantised exact geometry " << shape.name;
        EXPECT_EQ(observed_shapes, expected_shapes);

        EXPECT_EQ(M_VALUES, llaminar2::supportedPrefillGraphBucketSizes());
        EXPECT_EQ(std::count(M_VALUES.begin(), M_VALUES.end(), 1), 0);
        for (int m = 2; m <= llaminar2::kDefaultNativeQuantisedVerifierRowCapacity;
             ++m)
            EXPECT_EQ(std::count(M_VALUES.begin(), M_VALUES.end(), m), 0)
                << "M=" << m << " belongs to grouped verifier training";
    }

    /** @brief Direct native preparation publishes complete geometry for every source format. */
    TEST(NativeQuantisedSweepOffline, NativePreparationPublishesCompleteGeometry)
    {
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            SCOPED_TRACE(format.name);
            for (const auto [n, k] : {std::pair{65, 256}, std::pair{3, 512}})
            {
                auto weights = format.create(n, k);
                ROCmPackedWeights packed;
                packed.N = 17;
                packed.K = 32;
                ASSERT_TRUE(packNativeQuantised(weights.get(), packed));
                EXPECT_EQ(packed.N, n);
                EXPECT_EQ(packed.K, k);
                EXPECT_EQ(packed.native_quantised_blocks_per_row, k / 32);
                EXPECT_TRUE(packed.native_source_identity.present);
                EXPECT_EQ(packed.native_quantised_codebook_id,
                    canonicalDeviceVnniCodebookId(requireNativeQuantisedInfo(weights.get(), format.name)));
            }
        }
    }

    // =========================================================================
    // Variant definitions
    // =========================================================================

    /** @brief Exact producer controls, or null producer for installed Auto. */
    struct VariantConfig
    {
        std::string name;
        int force_n;    // -1=auto, 64=force N64, 128=force N128
        int mt;         // -1=auto, 16/32/64
        int min_blocks; // -1=auto, 1=bare, 2=2-wave, 3=3-wave
        int unroll;     // -1=auto, 0/1/2/4
        int full_tiles = 0; // -1=auto, 0=checked edges, 1=proven full tiles
        std::optional<VNNIPrefillProducer> producer =
            VNNIPrefillProducer::NativeCooperative;
    };

    static const std::vector<VariantConfig> NATIVE_QUANTISED_VARIANTS = {
        // N64 × M_TILE × MIN_BLOCKS sweep
        {"N64/MT16/MB1", 64, 16, 1, -1},
        {"N64/MT16/MB2", 64, 16, 2, -1},
        {"N64/MT32/MB1", 64, 32, 1, -1},
        {"N64/MT32/MB2", 64, 32, 2, -1},
        {"N64/MT64/MB1", 64, 64, 1, -1},
        {"N64/MT64/MB2", 64, 64, 2, -1},
        // N128 × M_TILE × MIN_BLOCKS sweep
        {"N128/MT16/MB1", 128, 16, 1, -1},
        {"N128/MT16/MB2", 128, 16, 2, -1},
        {"N128/MT32/MB1", 128, 32, 1, -1},
        {"N128/MT32/MB2", 128, 32, 2, -1},
        // UNROLL_G sweep on best known config (N64/MT64/MB1 — most likely winner)
        {"N64/MT64/MB1/U0", 64, 64, 1, 0},
        {"N64/MT64/MB1/U1", 64, 64, 1, 1},
        {"N64/MT64/MB1/U2", 64, 64, 1, 2},
        {"N64/MT64/MB1/U4", 64, 64, 1, 4},
        // The two-wave family is the production Q6_K contender. Exposing its
        // unroll dimension is essential: launch bounds and loop unrolling
        // jointly determine gfx906 VGPR allocation and scratch spills.
        {"N64/MT64/MB2/U0", 64, 64, 2, 0},
        {"N64/MT64/MB2/U1", 64, 64, 2, 1},
        {"N64/MT64/MB2/U2", 64, 64, 2, 2},
        {"N64/MT64/MB2/U4", 64, 64, 2, 4},
        // UNROLL_G sweep on N128/MT32/MB1
        {"N128/MT32/MB1/U0", 128, 32, 1, 0},
        {"N128/MT32/MB1/U1", 128, 32, 1, 1},
        {"N128/MT32/MB1/U2", 128, 32, 1, 2},
        {"N128/MT32/MB1/U4", 128, 32, 1, 4},
        {"N128/MT32/MB2/U0", 128, 32, 2, 0},
        {"N128/MT32/MB2/U1", 128, 32, 2, 1},
        {"N128/MT32/MB2/U2", 128, 32, 2, 2},
        {"N128/MT32/MB2/U4", 128, 32, 2, 4},
        // Auto dispatch (current heuristic)
        {"Auto", -1, -1, -1, -1, -1, std::nullopt},
    };

    /**
     * @brief Return the extra spill-gated Q6_K full-tile candidate family.
     *
     * Q6_K owns the dominant Qwen 3.6 prefill time and every released model
     * geometry is divisible by both candidate column tiles. M16 and M32 cover
     * the occupancy transition across the complete captured-bucket inventory;
     * M64 is deliberately absent because prior gfx906 ISA inspection proved
     * that its dual-scale accumulator shape spills.
     */
    static std::vector<VariantConfig> q6FullTileVariants()
    {
        std::vector<VariantConfig> result;
        for (const int n_tile : {64, 128})
        {
            for (const int m_tile : {16, 32})
            {
                for (const int min_blocks : {1, 2})
                {
                    for (const int unroll : {0, 1, 2, 4})
                    {
                        result.push_back(VariantConfig{
                            .name =
                                "N" + std::to_string(n_tile) +
                                "/MT" + std::to_string(m_tile) +
                                "/MB" + std::to_string(min_blocks) +
                                "/U" + std::to_string(unroll) +
                                "/FULL",
                            .force_n = n_tile,
                            .mt = m_tile,
                            .min_blocks = min_blocks,
                            .unroll = unroll,
                            .full_tiles = 1,
                        });
                    }
                }
            }
        }
        return result;
    }

    /**
     * @brief Return the exact candidate inventory launchable for one source format.
     * @param format_name Original source format, including independent prepared aliases.
     * @param prepared_family Explicit native preparation or the source-default contract.
     * @param device_ordinal Admitted visible participant whose native ISA owns candidate admission.
     * @return Concrete launch candidates and the existing Auto control for that contract.
     */
    static std::vector<VariantConfig> trainerVariantsForFormat(
        const std::string &format_name,
        std::optional<DensePrefillWeightFamily> prepared_family = std::nullopt,
        int device_ordinal = 0)
    {
        std::vector<VariantConfig> result;
        const auto append_wmma = [prepared_family, device_ordinal](std::vector<VariantConfig> &variants) {
            if (prepared_family != DensePrefillWeightFamily::NativeQuantised) return;
            if (nativeQuantisedHardwareForDevice(device_ordinal).architecture !=
                native_quantised::NativeQuantisedArchitecture::rocm(11, 0, 0)) return;
            for (const auto [mt, nt] : {std::pair{32, 32}, std::pair{32, 64}, std::pair{64, 64}, std::pair{64, 128}})
                variants.push_back({"WMMA/N" + std::to_string(nt) + "/MT" + std::to_string(mt),
                    nt, mt, 1, 0, 0, VNNIPrefillProducer::NativeWmma});
        };
        result.reserve(NATIVE_QUANTISED_VARIANTS.size() + 32);
        const auto auto_iterator = std::find_if(
            NATIVE_QUANTISED_VARIANTS.begin(),
            NATIVE_QUANTISED_VARIANTS.end(),
            [](const VariantConfig &variant)
            { return variant.name == "Auto"; });
        if (format_name == "Q8_0" || format_name == "Q8_1" || format_name == "Q8_K")
        {
            if (prepared_family == DensePrefillWeightFamily::NativeQuantised)
            {
                // Explicit original native K32 payload/scales. These three
                // existing physical configurations are the bounded prepared-Q8
                // comparison, not aliases for the expanded INT8 producers.
                result = {{"N64/MT16/MB1/U2",64,16,1,2,0,VNNIPrefillProducer::NativeCooperative},
                        {"N64/MT32/MB1/U2",64,32,1,2,0,VNNIPrefillProducer::NativeCooperative},
                        {"N128/MT32/MB1/U2",128,32,1,2,0,VNNIPrefillProducer::NativeCooperative},
                        *auto_iterator};
                append_wmma(result);
                return result;
            }
            // These controls belong to the actual INT8 blockwise producer.
            // NativeQuantised overrides have no effect on Q8 and must not create
            // fictitious distinct candidates with identical physical launches.
            for (const auto producer : {VNNIPrefillProducer::Int8BlockwiseV3,
                                        VNNIPrefillProducer::Int8BlockwiseV7})
                for (const int mt : {16, 32, 64})
                    for (const int unroll : {0, 1, 2, 4})
                    {
                        const bool v3 = producer == VNNIPrefillProducer::Int8BlockwiseV3;
                        result.push_back({
                            .name = std::string(v3 ? "INT8V3" : "INT8V7") +
                                "/MT" + std::to_string(mt) + "/U" + std::to_string(unroll),
                            .force_n = v3 ? 64 : 128, .mt = mt,
                            .min_blocks = 1, .unroll = unroll,
                            .full_tiles = 0, .producer = producer,
                        });
                    }
            result.push_back(*auto_iterator);
            return result;
        }
        for (auto iterator = NATIVE_QUANTISED_VARIANTS.begin();
             iterator != auto_iterator; ++iterator)
        {
            const bool q6_checked_spill =
                format_name == "Q6_K" &&
                iterator->full_tiles == 0 &&
                iterator->min_blocks == 2 &&
                ((iterator->force_n == 64 && iterator->mt == 64) ||
                 (iterator->force_n == 128 && iterator->mt == 32));
            if (!q6_checked_spill)
                result.push_back(*iterator);
        }
        if (format_name == "Q6_K")
        {
            auto full = q6FullTileVariants();
            result.insert(
                result.end(),
                std::make_move_iterator(full.begin()),
                std::make_move_iterator(full.end()));
        }
        if (auto_iterator != NATIVE_QUANTISED_VARIANTS.end())
            result.push_back(*auto_iterator);
        append_wmma(result);
        return result;
    }

    // =========================================================================
    // Benchmark result
    // =========================================================================

    struct BenchResult
    {
        std::string variant_name;
        std::string shape_name;
        std::string category;
        int M, N, K;
        double min_us = 0.0;
        double mean_us = 0.0;
        double stddev_us = 0.0;
        double gflops = 0.0;
        float cosine_sim = 0.0f;
        bool correctness_pass = false;
        size_t byte_mismatches_vs_auto = 0;
        size_t first_byte_mismatch_vs_auto =
            std::numeric_limits<size_t>::max();
        int observed_n_tile = 0;
        int observed_m_tile = 0;
        int observed_min_blocks = 0;
        int observed_unroll = 0;
        bool observed_full_tiles = false;
        VNNIPrefillProducer observed_producer = VNNIPrefillProducer::NativeCooperative;
        int registers_per_thread = 0;
        size_t local_memory_bytes_per_thread = 0;
        size_t static_shared_memory_bytes = 0;
        size_t dynamic_shared_memory_bytes = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;
    };

#ifdef HAVE_ROCM
    /**
     * @brief Typed, process-local NativeQuantised candidate override for the trainer.
     *
     * The production launcher reads its tuning controls from the typed
     * `DebugEnv` snapshot. Rewriting environment variables and reparsing the
     * complete ROCm configuration for every launch made the old trainer both
     * noisy and needlessly expensive. This scope changes only the five launch
     * fields owned by the selected producer and restores both producer families
     * on teardown. Auto clears all overrides rather than inheriting a prior
     * candidate's INT8 or NativeQuantised controls.
     * The turnkey collector gives every physical GPU its own process, so these
     * process-local controls cannot race another candidate lane.
     */
    class ScopedROCmNativeQuantisedTrainerOverride
    {
    public:
        ScopedROCmNativeQuantisedTrainerOverride()
        {
            saved_wmma_ = nativeQuantisedWmmaOverride();
            const auto &config = debugEnv().rocm;
            saved_gemv_kb_ = config.native_quantised_gemv_kb;
            saved_mt_ = config.native_quantised_mt;
            saved_unroll_ = config.native_quantised_unroll;
            saved_min_blocks_ = config.native_quantised_min_blocks;
            saved_force_n64_ = config.native_quantised_force_n64;
            saved_force_n128_ = config.native_quantised_force_n128;
            saved_full_tiles_ = config.native_quantised_full_tiles;
            saved_int8_v3_ = config.blockwise_force_v3;
            saved_int8_v7_ = config.blockwise_force_v7;
            saved_int8_v3_mt_ = config.blockwise_v3_mt;
            saved_int8_v7_mt_ = config.blockwise_v7_mt;
            saved_int8_v3_unroll_ = config.blockwise_v3_unroll;
            saved_int8_v7_unroll_ = config.blockwise_v7_unroll;
        }

        ~ScopedROCmNativeQuantisedTrainerOverride()
        {
            setNativeQuantisedWmmaOverride(saved_wmma_);
            auto &config = mutableDebugEnv().rocm;
            config.native_quantised_gemv_kb = saved_gemv_kb_;
            config.native_quantised_mt = saved_mt_;
            config.native_quantised_unroll = saved_unroll_;
            config.native_quantised_min_blocks = saved_min_blocks_;
            config.native_quantised_force_n64 = saved_force_n64_;
            config.native_quantised_force_n128 = saved_force_n128_;
            config.native_quantised_full_tiles = saved_full_tiles_;
            config.blockwise_force_v3 = saved_int8_v3_;
            config.blockwise_force_v7 = saved_int8_v7_;
            config.blockwise_v3_mt = saved_int8_v3_mt_;
            config.blockwise_v7_mt = saved_int8_v7_mt_;
            config.blockwise_v3_unroll = saved_int8_v3_unroll_;
            config.blockwise_v7_unroll = saved_int8_v7_unroll_;
        }

        ScopedROCmNativeQuantisedTrainerOverride(
            const ScopedROCmNativeQuantisedTrainerOverride &) = delete;
        ScopedROCmNativeQuantisedTrainerOverride &operator=(
            const ScopedROCmNativeQuantisedTrainerOverride &) = delete;

        /** Select one exact compile-time candidate for subsequent launches. */
        void apply(const VariantConfig &variant)
        {
            std::optional<NativeQuantisedWmmaTile> wmma;
            if (variant.producer == VNNIPrefillProducer::NativeWmma)
                wmma = variant.force_n == 128 ? NativeQuantisedWmmaTile::M64N128 :
                    variant.force_n == 32 ? NativeQuantisedWmmaTile::M32N32 :
                    variant.mt == 32 ? NativeQuantisedWmmaTile::M32N64 : NativeQuantisedWmmaTile::M64N64;
            setNativeQuantisedWmmaOverride(wmma);
            auto &config = mutableDebugEnv().rocm;
            const bool native = variant.producer == VNNIPrefillProducer::NativeCooperative;
            const bool int8_v3 = variant.producer == VNNIPrefillProducer::Int8BlockwiseV3;
            const bool int8_v7 = variant.producer == VNNIPrefillProducer::Int8BlockwiseV7;
            config.native_quantised_mt = native ? variant.mt : -1;
            config.native_quantised_unroll = native ? variant.unroll : -1;
            config.native_quantised_min_blocks = native ? variant.min_blocks : -1;
            config.native_quantised_force_n64 = native && variant.force_n == 64;
            config.native_quantised_force_n128 = native && variant.force_n == 128;
            config.native_quantised_full_tiles = native ? variant.full_tiles : -1;
            config.blockwise_force_v3 = int8_v3;
            config.blockwise_force_v7 = int8_v7;
            config.blockwise_v3_mt = int8_v3 ? variant.mt : -1;
            config.blockwise_v7_mt = int8_v7 ? variant.mt : -1;
            config.blockwise_v3_unroll = int8_v3 ? variant.unroll : -1;
            config.blockwise_v7_unroll = int8_v7 ? variant.unroll : -1;
        }

    private:
        std::optional<NativeQuantisedWmmaTile> saved_wmma_;
        int saved_gemv_kb_ = -1;
        int saved_mt_ = -1;
        int saved_unroll_ = -1;
        int saved_min_blocks_ = -1;
        bool saved_force_n64_ = false;
        bool saved_force_n128_ = false;
        int saved_full_tiles_ = -1;
        bool saved_int8_v3_ = false;
        bool saved_int8_v7_ = false;
        int saved_int8_v3_mt_ = -1;
        int saved_int8_v7_mt_ = -1;
        int saved_int8_v3_unroll_ = -1;
        int saved_int8_v7_unroll_ = -1;
    };

    /** @brief Move-only ownership for one HIP timing event. */
    class ScopedROCmTimingEvent
    {
    public:
        ScopedROCmTimingEvent()
        {
            const hipError_t error =
                hipEventCreateWithFlags(&event_, hipEventDefault);
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("failed to create HIP timing event: ") +
                    hipGetErrorString(error));
            }
        }

        ~ScopedROCmTimingEvent()
        {
            if (event_)
                (void)hipEventDestroy(event_);
        }

        ScopedROCmTimingEvent(const ScopedROCmTimingEvent &) = delete;
        ScopedROCmTimingEvent &operator=(const ScopedROCmTimingEvent &) = delete;

        ScopedROCmTimingEvent(ScopedROCmTimingEvent &&other) noexcept
            : event_(other.event_)
        {
            other.event_ = nullptr;
        }

        ScopedROCmTimingEvent &operator=(ScopedROCmTimingEvent &&other) noexcept
        {
            if (this != &other)
            {
                if (event_)
                    (void)hipEventDestroy(event_);
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }

        [[nodiscard]] hipEvent_t get() const noexcept { return event_; }

    private:
        hipEvent_t event_ = nullptr;
    };

    /** @brief Explicit timing surface; diagnostic graph replay is not a training label. */
    enum class SweepSubmission
    {
        Direct, ///< Candidate tournament; retain historical timing semantics.
        Captured, ///< Projection pipeline, including activation quantization.
        CapturedProducer ///< GEMM alone, consuming the oracle's prepared Q8 activations.
    };

    /** @brief Construction policy; generic timing never measures installed exact entries. */
    enum class SweepDispatch { GenericTournament, InstalledProduction };

    /**
     * @brief Scoped construction-thread lookup policy with nested oracle support.
     * @param dispatch The policy to retain for this diagnostic object's lifetime.
     * Existing graphs contain their exact producer and never consult this switch.
     */
    class ScopedDensePrefillLookup
    {
    public:
        /** @brief Retain an explicit diagnostic lookup policy on this construction thread. */
        explicit ScopedDensePrefillLookup(SweepDispatch dispatch)
            : previous_(densePrefillExactOverlayEnabled())
        {
            setDensePrefillExactOverlayEnabled(dispatch == SweepDispatch::InstalledProduction);
        }
        /** @brief Restore the preceding construction policy, including on failure. */
        ~ScopedDensePrefillLookup() { setDensePrefillExactOverlayEnabled(previous_); }
        ScopedDensePrefillLookup(const ScopedDensePrefillLookup &) = delete;
        ScopedDensePrefillLookup &operator=(const ScopedDensePrefillLookup &) = delete;
    private:
        bool previous_;
    };

    /** @brief Production dispatcher with independent physical and source-math identities. */
    extern "C" bool rocmGemm_native_quantised_fp32_with_policy(
        const int8_t *, const uint8_t *, const void *, const void *, const void *,
        float *, const float *, const float *, int, int, int, uint8_t, uint8_t,
        int, void *);

    /**
     * @brief Persistent device state for one ROCm shape/M tournament.
     *
     * Construction performs weight packing/upload, tensor allocation, input
     * publication, workspace planning, event creation, and the exact-M Auto
     * oracle launch exactly once. `measure()` then contains only warmup and
     * timed production kernel launches on one explicit non-default stream.
     * `certify()` compares every output byte on that producer stream and
     * downloads only the mismatch count and first mismatch offset.
     *
     * Candidate geometry changes do not change any captured pointer or arena
     * binding. The workspace is the union of every candidate requirement for
     * this cell, making dynamic allocation or rebinding during measurement
     * structurally unnecessary.
     */
    class PreparedROCmSweepExecution
    {
    public:
        /**
         * @brief Prepare one original source ABI and its retained exact-M oracle.
         * @param shape Immutable N/K geometry retained throughout this execution object's lifetime.
         * @param m Positive live row count, shared by every candidate and its oracle.
         * @param weights Original host-owned source weights, packed once during construction.
         * @param variants Complete candidate set whose workspace requirements are admitted together.
         * @param maximum_bench_runs Positive capacity for persistent timing events and samples.
         * @param dispatch Explicit generic tournament or installed production lookup policy.
         * @param prepared_family Null preserves source-default preparation; the
         * explicit NativeQuantised value preserves each source format's native payload.
         * The choice is immutable throughout this invocation and capture.
         * @param device_ordinal Original visible ordinal owning every retained operand and graph.
         * @throws std::invalid_argument On incomplete geometry or unsupported preparation policy.
         * @throws std::runtime_error On failed device preparation, allocation or oracle publication.
         */
        PreparedROCmSweepExecution(
            const GEMMShape &shape,
            int m,
            TensorBase *weights,
            const std::vector<VariantConfig> &variants,
            int maximum_bench_runs,
            SweepDispatch dispatch = SweepDispatch::GenericTournament,
            std::optional<DensePrefillWeightFamily> prepared_family = std::nullopt,
            int device_ordinal = 0)
            : device_ordinal_(device_ordinal), overlay_lookup_(dispatch), shape_(shape), m_(m)
        {
            if (!weights || m_ <= 0 || shape_.N <= 0 || shape_.K <= 0 ||
                maximum_bench_runs <= 0)
            {
                throw std::invalid_argument(
                    "invalid ROCm NativeQuantised tournament geometry");
            }
            requireHip(hipSetDevice(device_ordinal_), "select device");
            expected_codebook_ = canonicalDeviceVnniCodebookId(
                requireNativeQuantisedInfo(weights, "ROCm tournament source weights"));
            if (prepared_family && *prepared_family != DensePrefillWeightFamily::NativeQuantised)
                throw std::invalid_argument("Explicit sweep preparation requires original native payloads");
            const bool packed = prepared_family
                ? packNativeQuantised(weights, packed_)
                : packWeightsToROCm(weights, packed_);
            if (!packed)
                throw std::runtime_error(
                    "failed to pack persistent ROCm NativeQuantised weights");

            kernel_ = std::make_unique<ROCmQuantisedGemmKernel>(
                &packed_, device_ordinal_);
            stream_owner_ = std::make_unique<ScopedGPUStream>(
                DeviceId::rocm(device_ordinal_));
            stream_ = static_cast<hipStream_t>(stream_owner_->get());
            kernel_->setGPUStream(static_cast<void *>(stream_));

            WorkspaceRequirements requirements;
            for (const auto &variant : variants)
            {
                override_.apply(variant);
                requirements.merge(kernel_->getWorkspaceRequirements(
                    m_, shape_.N, shape_.K));
            }
            override_.apply(autoVariant());
            requirements.merge(kernel_->getWorkspaceRequirements(
                m_, shape_.N, shape_.K));

            const size_t required = requirements.total_bytes_with_alignment();
            const size_t budget = std::max(
                required + required / 10,
                size_t{64} * 1024 * 1024);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_ordinal_), budget);
            if (!workspace_->allocate(requirements))
                throw std::runtime_error(
                    "failed to allocate persistent ROCm sweep workspace");
            kernel_->bindWorkspace(workspace_.get());

            input_ = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(m_), static_cast<size_t>(shape_.K)},
                -0.25f,
                0.25f,
                7);
            output_ = TestTensorFactory::createFP32(
                {static_cast<size_t>(m_), static_cast<size_t>(shape_.N)});
            oracle_output_ = TestTensorFactory::createFP32(
                {static_cast<size_t>(m_), static_cast<size_t>(shape_.N)});
            comparison_state_ = TestTensorFactory::createFP32({4});
            if (!input_ || !output_ || !oracle_output_ || !comparison_state_)
                throw std::runtime_error(
                    "failed to construct persistent ROCm sweep tensors");

            const DeviceId device = DeviceId::rocm(device_ordinal_);
            if (!input_->ensureOnDevice(device, stream_) ||
                !output_->allocateOnDevice(device, stream_) ||
                !oracle_output_->allocateOnDevice(device, stream_) ||
                !comparison_state_->allocateOnDevice(device, stream_))
            {
                throw std::runtime_error(
                    "failed to bind persistent ROCm sweep tensors");
            }

            start_events_.reserve(static_cast<size_t>(maximum_bench_runs));
            stop_events_.reserve(static_cast<size_t>(maximum_bench_runs));
            for (int index = 0; index < maximum_bench_runs; ++index)
            {
                start_events_.emplace_back();
                stop_events_.emplace_back();
            }
            times_us_.resize(static_cast<size_t>(maximum_bench_runs));

            buildExactMPrefillOracle();
        }

        ~PreparedROCmSweepExecution()
        {
            if (stream_)
                (void)hipStreamSynchronize(stream_);
            if (kernel_)
                kernel_->clearGPUStreamBinding();
            if (kernel_)
                kernel_->unbindWorkspace();
        }

        PreparedROCmSweepExecution(const PreparedROCmSweepExecution &) = delete;
        PreparedROCmSweepExecution &operator=(
            const PreparedROCmSweepExecution &) = delete;

        /**
         * @brief Time one candidate without allocation or per-launch blocking.
         *
         * Every event pair is enqueued first. Waiting on the final stop event
         * drains the ordered stream once, after which all individual elapsed
         * intervals are available. This preserves per-launch samples without
         * the old host round trip after every kernel invocation.
         */
        BenchResult measure(
            const VariantConfig &variant,
            int warmup_runs,
            int bench_runs,
            SweepSubmission submission = SweepSubmission::Direct)
        {
            if (warmup_runs <= 0 || bench_runs <= 0 ||
                static_cast<size_t>(bench_runs) > start_events_.size())
            {
                throw std::invalid_argument(
                    "invalid ROCm NativeQuantised timing repetition count");
            }

            override_.apply(variant);
            if (submission == SweepSubmission::CapturedProducer)
            {
                DeviceNativeQuantisedMatrixDesc weights;
                if (!kernel_->exportNativeQuantisedMatrixDesc(weights) || !weights.valid())
                    throw std::runtime_error(
                        "Producer-only NativeQuantised probe requires a NativeQuantised matrix; "
                        "This prepared family uses the separate expanded INT8 producer");
            }
            BenchResult result = inspectCandidate(variant, submission);

            // Capture after all persistent resources and launch metadata exist.
            // Production tournaments and scaling probes measure the retained
            // graph path used by inference. Independent complete projections
            // amortize only the outer host submission and timing-event cost.
            const int operations_per_sample = submission == SweepSubmission::Direct
                ? 1 : llaminar2::test::kDensePrefillCapturedOperations;
            std::unique_ptr<HIPGraphCapture> graph;
            if (submission != SweepSubmission::Direct)
            {
                graph = std::make_unique<HIPGraphCapture>(stream_, device_ordinal_);
                ScopedBackendGraphCapture capture(*graph, "NativeQuantised captured shape probe");
                if (!capture.begin())
                    throw std::runtime_error("cannot begin captured shape probe");
                for (int operation = 0; operation < operations_per_sample; ++operation)
                {
                    if (submission == SweepSubmission::CapturedProducer)
                        launchProducer();
                    else
                        launch(output_.get());
                }
                capture.finish();
                if (!graph->instantiate())
                    throw std::runtime_error("cannot instantiate captured shape probe");
            }
            const auto submit = [&]() {
                if (graph)
                {
                    if (!graph->launch())
                        throw std::runtime_error("cannot replay captured shape probe");
                }
                else
                    launch(output_.get());
            };
            for (int iteration = 0; iteration < warmup_runs; ++iteration)
                submit();
            checkStream("candidate warmup");
            if (graph)
                proveDirtyReplay(*graph, submission, result.observed_producer);

            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                const size_t index = static_cast<size_t>(iteration);
                requireHip(
                    hipEventRecord(start_events_[index].get(), stream_),
                    "record candidate start event");
                submit();
                requireHip(
                    hipEventRecord(stop_events_[index].get(), stream_),
                    "record candidate stop event");
            }
            requireHip(
                hipEventSynchronize(
                    stop_events_[static_cast<size_t>(bench_runs - 1)].get()),
                "wait for candidate event batch");

            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                float elapsed_ms = 0.0f;
                requireHip(
                    hipEventElapsedTime(
                        &elapsed_ms,
                        start_events_[static_cast<size_t>(iteration)].get(),
                        stop_events_[static_cast<size_t>(iteration)].get()),
                    "read candidate elapsed time");
                times_us_[static_cast<size_t>(iteration)] =
                    static_cast<double>(elapsed_ms) * 1000.0 / operations_per_sample;
            }

            result.min_us = *std::min_element(
                times_us_.begin(),
                times_us_.begin() + bench_runs);
            result.mean_us = std::accumulate(
                                 times_us_.begin(),
                                 times_us_.begin() + bench_runs,
                                 0.0) /
                             static_cast<double>(bench_runs);
            double squared_error = 0.0;
            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                const double sample = times_us_[static_cast<size_t>(iteration)];
                const double difference = sample - result.mean_us;
                squared_error += difference * difference;
            }
            result.stddev_us = std::sqrt(
                squared_error / static_cast<double>(bench_runs));
            const double flops =
                2.0 * static_cast<double>(m_) * shape_.N * shape_.K;
            result.gflops = flops / (result.min_us * 1.0e-6) / 1.0e9;

            const auto certificate = certify();
            result.byte_mismatches_vs_auto =
                static_cast<size_t>(certificate[0]);
            result.first_byte_mismatch_vs_auto =
                static_cast<size_t>(certificate[1]);
            result.correctness_pass = certificate[0] == 0;
            result.cosine_sim = result.correctness_pass ? 1.0f : 0.0f;
            last_timing_sample_count_ = static_cast<size_t>(bench_runs);
            return result;
        }

        /**
         * @brief Return the native HIP-event samples from the last candidate.
         *
         * The view aliases persistent tournament storage and is consumed
         * before another candidate launch. Keeping this allocation-free makes
         * the sidecar faithful to the same event batch used by the aggregate.
         */
        [[nodiscard]] std::span<const double> timingSamples() const noexcept
        {
            return std::span<const double>(
                times_us_.data(), last_timing_sample_count_);
        }

        /**
         * @brief Functional captured-replay proof, without a benchmark loop.
         * @param variant Exact producer to authenticate and byte-compare.
         * @param submission Complete projection or explicit production GEMM leaf.
         * @return Physical launch and complete-output byte-certificate evidence.
         * Poisoning the destination prevents the resource probe's prior output
         * from disguising a missing captured write. No timing threshold or
         * performance measurement belongs in this preflight operation.
         */
        BenchResult verifyCaptured(const VariantConfig &variant,
            SweepSubmission submission = SweepSubmission::Captured)
        {
            if (submission == SweepSubmission::Direct)
                throw std::invalid_argument("captured replay requires an explicit captured surface");
            BenchResult result = inspectCandidate(variant, submission);
            requireHip(hipMemsetAsync(output_->gpu_data_ptr(), 0xff,
                static_cast<size_t>(m_) * shape_.N * sizeof(float), stream_),
                "poison captured proof destination");
            HIPGraphCapture graph(stream_, device_ordinal_);
            ScopedBackendGraphCapture capture(graph, "VNNI producer identity proof");
            if (!capture.begin())
                throw std::runtime_error("cannot capture VNNI producer identity proof");
            if (submission == SweepSubmission::CapturedProducer) launchProducer();
            else launch(output_.get());
            capture.finish();
            if (!graph.instantiate())
                throw std::runtime_error("cannot instantiate VNNI producer identity proof");
            proveDirtyReplay(graph, submission, result.observed_producer);
            const auto certificate = certify();
            result.byte_mismatches_vs_auto = certificate[0];
            result.first_byte_mismatch_vs_auto = certificate[1];
            result.correctness_pass = certificate[0] == 0;
            return result;
        }

        /** @brief Profile one exact complete projection through retained production capture.
         * @param variant Exact physical candidate, selected from the existing trainer inventory.
         * @param request_id Immutable profiler request bound to sealed unprofiled observations.
         * @param warmups Untimed retained replay count before hardware-counter collection.
         * @throws std::runtime_error For resource, capture, byte or profiler-control failures.
         * The owning stream is explicit throughout. Oracle preparation and dirty
         * byte checks stay outside the ROCTx range; only one graph launch is resumed.
         */
        void profileExactCandidate(const VariantConfig &variant, const std::string &request_id, int warmups)
        {
            if (request_id.empty() || warmups <= 0)
                throw std::invalid_argument("ROCm dense profiler needs one complete exact request");
            const auto observed = inspectCandidate(variant);
            const std::string physical = std::to_string(observed.observed_n_tile) + "," +
                std::to_string(observed.observed_m_tile) + "," + std::to_string(observed.observed_min_blocks) + "," +
                std::to_string(observed.observed_unroll) + "," + (observed.observed_full_tiles ? "1" : "0") + "," +
                std::to_string(static_cast<int>(observed.observed_producer));
            if (physical != getEnvString("LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_EXPECTED_LAUNCH"))
                throw std::runtime_error("native prefill profiler physical launch differs from sealed timing");
            HIPGraphCapture graph(stream_, device_ordinal_);
            ScopedBackendGraphCapture capture(graph, "ROCm dense exact-candidate profiler");
            if (!capture.begin()) throw std::runtime_error("cannot capture ROCm dense profiler projection");
            launch(output_.get());
            capture.finish();
            if (!graph.instantiate()) throw std::runtime_error("cannot instantiate ROCm dense profiler projection");
            for (int replay = 0; replay < warmups; ++replay)
                if (!graph.launch()) throw std::runtime_error("ROCm dense profiler warmup failed");
            requireHip(hipMemsetAsync(output_->gpu_data_ptr(), 0xff,
                static_cast<size_t>(m_) * shape_.N * sizeof(float), stream_), "poison profiler precondition");
            if (!graph.launch() || certify()[0] != 0)
                throw std::runtime_error("ROCm dense profiler dirty replay changed exact-M output bytes");
            requireHip(hipMemsetAsync(output_->gpu_data_ptr(), 0xff,
                static_cast<size_t>(m_) * shape_.N * sizeof(float), stream_), "poison collected replay destination");
            checkStream("settle exact profiler preconditioning");
            const std::string range = "NativeQuantisedProfile::" + request_id;
            if (roctxRangePushA(range.c_str()) < 0)
                throw std::runtime_error("cannot open ROCm dense profiler range");
            if (roctxProfilerResume(0) != 0)
            {
                (void)roctxRangePop();
                throw std::runtime_error("cannot resume ROCm dense profiler collection");
            }
            const bool launched = graph.launch();
            const hipError_t completion = hipStreamSynchronize(stream_);
            const int paused = roctxProfilerPause(0);
            const int range_level = roctxRangePop();
            if (!launched || completion != hipSuccess || paused != 0 || range_level < 0)
                throw std::runtime_error("isolated ROCm dense profiler replay failed");
            if (certify()[0] != 0)
                throw std::runtime_error("collected ROCm dense replay changed exact-M output bytes");
            std::fprintf(stderr,
                "[NativeQuantisedProfiler][ROCm-Dense-Prefill] request=%s candidate=%s M=%d launches=1\n",
                request_id.c_str(), variant.name.c_str(), m_);
        }

        /**
         * @brief Prove the shared certificate rejects missing native graph work.
         * @param variant Exact launch whose actual quantizer and projection nodes are disabled.
         * Re-enable the same executable and require complete dirty replay after each fault.
         */
        void verifyDirtyReplayNegativeControls(const VariantConfig &variant)
        {
            const auto observed = inspectCandidate(variant);
            HIPGraphCapture graph(stream_, device_ordinal_);
            ScopedBackendGraphCapture capture(graph, "ROCm dirty replay controls");
            ASSERT_TRUE(capture.begin());
            launch(output_.get());
            capture.finish();
            ASSERT_TRUE(graph.instantiate());
            size_t count = 0;
            ASSERT_EQ(hipGraphGetNodes(graph.graph(), nullptr, &count), hipSuccess);
            std::vector<hipGraphNode_t> nodes(count);
            ASSERT_EQ(hipGraphGetNodes(graph.graph(), nodes.data(), &count), hipSuccess);
            for (const bool omit_quantizer : {false, true})
            {
                proveDirtyReplay(graph, SweepSubmission::Captured, observed.observed_producer);
                std::vector<hipGraphNode_t> disabled;
                int quantizers = 0, primaries = 0;
                for (const auto node : nodes)
                {
                    hipGraphNodeType type;
                    ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
                    if (type != hipGraphNodeTypeKernel) continue;
                    hipKernelNodeParams parameters{};
                    ASSERT_EQ(hipGraphKernelNodeGetParams(node, &parameters), hipSuccess);
                    ASSERT_NE(parameters.kernelParams, nullptr);
                    ASSERT_NE(parameters.kernelParams[0], nullptr);
                    const void *first_argument = *static_cast<void **>(parameters.kernelParams[0]);
                    const bool quantizer = first_argument == input_->gpu_data_ptr();
                    const bool primary = first_argument == workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A);
                    ASSERT_TRUE(quantizer || primary) << "unrecognized captured kernel role";
                    quantizers += quantizer; primaries += primary;
                    if (quantizer != omit_quantizer) continue;
                    ASSERT_EQ(hipGraphNodeSetEnabled(graph.executable(), node, 0), hipSuccess);
                    disabled.push_back(node);
                }
                ASSERT_EQ(quantizers, 1);
                ASSERT_EQ(primaries, 1);
                ASSERT_FALSE(disabled.empty());
                EXPECT_THROW(proveDirtyReplay(graph, SweepSubmission::Captured,
                    observed.observed_producer), std::runtime_error);
                for (const auto node : disabled)
                    ASSERT_EQ(hipGraphNodeSetEnabled(graph.executable(), node, 1), hipSuccess);
                EXPECT_NO_THROW(proveDirtyReplay(graph, SweepSubmission::Captured,
                    observed.observed_producer));
            }
        }

    private:
        const int device_ordinal_;

        /**
         * @brief Poison the exact live banks consumed by this retained projection.
         * @param poison Reset byte pattern, outside timing and capture.
         * @param submission Producer-only probes preserve their authenticated quantized inputs.
         * @param producer Observed native or expanded-INT8 physical producer.
         */
        void poisonProjection(int poison, SweepSubmission submission, VNNIPrefillProducer producer)
        {
            const auto bank = [&](const char *name, size_t bytes) {
                if (!workspace_->getBuffer(name) || workspace_->getBufferSize(name) < bytes)
                    throw std::runtime_error("ROCm dirty replay scratch extent exceeds admission");
                requireHip(hipMemsetAsync(workspace_->getBuffer(name), poison, bytes, stream_),
                    "poison consumed projection scratch");
            };
            const size_t output_bytes = static_cast<size_t>(m_) * shape_.N * sizeof(float);
            requireHip(hipMemsetAsync(output_->gpu_data_ptr(), poison, output_bytes, stream_),
                "poison projection destination");
            if (submission != SweepSubmission::CapturedProducer)
            {
                bank(GemmWorkspaceBuffers::QUANT_A, static_cast<size_t>(m_) * shape_.K);
                bank(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
                    static_cast<size_t>(m_) * (shape_.K / 32) * sizeof(float));
            }
            if (producer == VNNIPrefillProducer::Int8BlockwiseV3 ||
                producer == VNNIPrefillProducer::Int8BlockwiseV7)
                bank(GemmWorkspaceBuffers::ACC_INT32, output_bytes);
        }

        /**
         * @brief Certify three dirty resets of the actual retained executable.
         * @param graph Executable whose nodes will also supply the timing samples.
         * @param submission Complete projection or authenticated producer-only surface.
         * @param producer Exact producer observed before recording.
         * @throws std::runtime_error On omitted work or any changed output byte.
         */
        void proveDirtyReplay(HIPGraphCapture &graph, SweepSubmission submission,
                              VNNIPrefillProducer producer)
        {
            for (const int poison : {0xff, 0x55, 0x00})
            {
                poisonProjection(poison, submission, producer);
                if (!graph.launch()) throw std::runtime_error("ROCm dirty replay launch failed");
                if (certify()[0] != 0)
                    throw std::runtime_error("ROCm dirty replay changed exact-M output bytes");
            }
        }

        /**
         * @brief Resolve physical identity and compiler resources before timing.
         * @param variant Requested controls, or installed Auto.
         * @param submission Physical projection or explicitly isolated GEMM leaf.
         * @return Complete physical observation; stale/foreign/spilling launches
         *         throw before any timing or artifact can be published.
         */
        BenchResult inspectCandidate(const VariantConfig &variant,
            SweepSubmission submission = SweepSubmission::Direct)
        {
            override_.apply(variant);
            clearVNNIPrefillLaunch();
            if (submission == SweepSubmission::CapturedProducer) launchProducer();
            else launch(output_.get());
            checkStream("candidate resource probe");
            const auto observed = lastVNNIPrefillLaunch();
            VNNIPrefillLaunchResources resources;
            if (!observed || observed->execution_codebook != expected_codebook_ ||
                (variant.producer && observed->producer != *variant.producer) ||
                !inspectVNNIPrefillLaunch(*observed, resources))
                throw std::runtime_error("ROCm VNNI trainer could not inspect " + variant.name +
                    " M=" + std::to_string(m_) + " N=" + std::to_string(shape_.N) +
                    " K=" + std::to_string(shape_.K));
            BenchResult result;
            result.variant_name = variant.name;
            result.shape_name = shape_.name;
            result.category = shape_.category;
            result.M = m_;
            result.N = shape_.N;
            result.K = shape_.K;
            result.observed_producer = observed->producer;
            result.observed_n_tile = observed->n_tile;
            result.observed_m_tile = observed->m_tile;
            result.observed_min_blocks = observed->min_blocks;
            result.observed_unroll = observed->unroll;
            result.observed_full_tiles = observed->full_tiles;
            result.dynamic_shared_memory_bytes = observed->dynamic_shared_memory_bytes;
            result.registers_per_thread = resources.registers_per_thread;
            result.local_memory_bytes_per_thread = resources.local_memory_bytes_per_thread;
            result.static_shared_memory_bytes = resources.static_shared_memory_bytes;
            result.max_threads_per_block = resources.max_threads_per_block;
            result.max_active_blocks_per_sm = resources.max_active_blocks_per_sm;
            if (result.local_memory_bytes_per_thread != 0)
                throw std::runtime_error("ROCm VNNI candidate spills local memory: " +
                    variant.name + " bytes_per_thread=" +
                    std::to_string(result.local_memory_bytes_per_thread));
            return result;
        }

        /** Return the production exact-M route used as the byte oracle. */
        static const VariantConfig &autoVariant()
        {
            const auto iterator = std::find_if(
                NATIVE_QUANTISED_VARIANTS.begin(),
                NATIVE_QUANTISED_VARIANTS.end(),
                [](const VariantConfig &variant)
                { return variant.name == "Auto"; });
            if (iterator == NATIVE_QUANTISED_VARIANTS.end())
                throw std::logic_error(
                    "ROCm NativeQuantised candidate inventory has no Auto route");
            return *iterator;
        }

        static void requireHip(hipError_t error, const char *operation)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("ROCm NativeQuantised sweep failed to ") +
                    operation + ": " + hipGetErrorString(error));
            }
        }

        void launch(FP32Tensor *destination)
        {
            if (!kernel_->multiply_tensor(
                    input_.get(),
                    destination,
                    m_,
                    shape_.N,
                    shape_.K))
            {
                throw std::runtime_error(
                    "ROCm NativeQuantised production launch failed");
            }
        }

        /**
         * @brief Isolate GEMM without confusing replicated quantization with TP scaling.
         *
         * The untimed production oracle already populated these persistent
         * activation buffers. Reuse exactly that representation and its source
         * arithmetic policy, then byte-certify the resulting projection against
         * the complete production pipeline. No alternate arithmetic, packing,
         * workspace allocation or dispatcher is introduced by the probe.
         */
        void launchProducer()
        {
            DeviceNativeQuantisedMatrixDesc weights;
            const bool exported = kernel_->exportNativeQuantisedMatrixDesc(weights);
            if (!exported || !weights.valid() ||
                !weights.source_identity_present)
                throw std::runtime_error("captured producer requires prepared source-aware weights: M=" +
                    std::to_string(m_) + " expected_codebook=" + std::to_string(expected_codebook_) +
                    " exported=" + std::to_string(exported) + " n=" + std::to_string(weights.n) +
                    " k=" + std::to_string(weights.k) + " blocks=" + std::to_string(weights.blocks_per_row) +
                    " source_present=" + std::to_string(weights.source_identity_present));
            const auto *activations = static_cast<const int8_t *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            const auto *row_scales = static_cast<const float *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
            const auto *block_scales = static_cast<const float *>(
                workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            if (!activations || !row_scales || !block_scales ||
                !rocmGemm_native_quantised_fp32_with_policy(
                    activations, weights.payload, weights.scales, weights.mins, weights.emins,
                    static_cast<float *>(output_->gpu_data_ptr()), row_scales, block_scales,
                    m_, shape_.N, shape_.K, weights.codebook_id,
                    canonicalDeviceVnniCodebookId(weights.source_codebook_id), device_ordinal_, stream_))
                throw std::runtime_error("captured NativeQuantised producer launch failed");
        }

        void checkStream(const char *operation)
        {
            requireHip(hipStreamSynchronize(stream_), operation);
        }

        void buildExactMPrefillOracle()
        {
            ScopedDensePrefillLookup generic_oracle(SweepDispatch::GenericTournament);
            override_.apply(autoVariant());
            launch(oracle_output_.get());
            checkStream("construct exact-M Auto oracle");
        }

        std::array<uint64_t, 2> certify()
        {
            auto *comparison_words = reinterpret_cast<uint64_t *>(
                comparison_state_->gpu_data_ptr());
            if (!llaminar2::test::enqueueROCmFP32ByteComparison(
                    reinterpret_cast<const float *>(output_->gpu_data_ptr()),
                    reinterpret_cast<const float *>(
                        oracle_output_->gpu_data_ptr()),
                    static_cast<size_t>(m_) *
                        static_cast<size_t>(shape_.N),
                    comparison_words,
                    comparison_words + 1,
                    stream_))
            {
                throw std::runtime_error(
                    "ROCm NativeQuantised full-buffer byte comparison failed");
            }

            std::array<uint64_t, 2> certificate{};
            requireHip(
                hipMemcpyAsync(
                    certificate.data(),
                    comparison_words,
                    sizeof(certificate),
                    hipMemcpyDeviceToHost,
                    stream_),
                "download byte-certificate counters");
            checkStream("complete byte certificate");
            return certificate;
        }

        ScopedDensePrefillLookup overlay_lookup_;
        const GEMMShape &shape_;
        int m_ = 0;
        uint8_t expected_codebook_ = 0;
        ScopedROCmNativeQuantisedTrainerOverride override_;
        std::unique_ptr<ScopedGPUStream> stream_owner_;
        hipStream_t stream_ = nullptr;
        ROCmPackedWeights packed_;
        std::unique_ptr<ROCmQuantisedGemmKernel> kernel_;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        std::unique_ptr<FP32Tensor> input_;
        std::unique_ptr<FP32Tensor> output_;
        std::unique_ptr<FP32Tensor> oracle_output_;
        std::unique_ptr<FP32Tensor> comparison_state_;
        std::vector<ScopedROCmTimingEvent> start_events_;
        std::vector<ScopedROCmTimingEvent> stop_events_;
        std::vector<double> times_us_;
        size_t last_timing_sample_count_ = 0;
    };
#endif

    // =========================================================================
    // Test fixture
    // =========================================================================

    /** @brief Publish original device facts before receipts or prepared operands consume them. */
    class NativeQuantisedSweepTest : public ::testing::Test
    {
    protected:
        /** @brief Admit the canonical inventory without relying on a later tensor upload. */
        void SetUp() override
        {
#ifdef HAVE_ROCM
            initializeROCmHardwareMetadata();
            const auto &inventory = rocmHardwareMetadata();
            has_device_ = inventory.size() != 0;
            if (has_device_)
            {
                const auto &facts = inventory.at(0);
                device_name_ = facts.name + " (" + facts.isa_target + ")";
            }
            for (size_t index = 0; index < inventory.size(); ++index)
            {
                const auto &facts = inventory.at(index);
                if (native_quantised::NativeQuantisedArchitecture::fromROCmISATarget(facts.isa_target) ==
                    native_quantised::NativeQuantisedArchitecture::rocm(11, 0, 0))
                {
                    wmma_device_ordinal_ = facts.ordinal;
                    break;
                }
            }
#else
            has_device_ = false;
#endif
        }

        bool has_device_ = false;
        std::string device_name_;
        std::optional<int> wmma_device_ordinal_;

        /** @brief Bind replay evidence to the admitted native participant, without rediscovery. */
        void recordWmmaDevice() const
        {
            const auto &facts = rocmHardwareMetadata().at(wmma_device_ordinal_.value());
            std::string uuid;
            constexpr char digits[] = "0123456789abcdef";
            for (const auto byte : facts.uuid)
            {
                uuid += digits[byte >> 4];
                uuid += digits[byte & 15];
            }
            RecordProperty("wmma_device_ordinal", facts.ordinal);
            RecordProperty("wmma_device_uuid", uuid);
            RecordProperty("wmma_device_isa", facts.isa_target);
        }
    };

    /** @brief A fresh filtered trainer publishes actual identity before preparing any tensor. */
    TEST_F(NativeQuantisedSweepTest, NativeQ8MetadataAvailableBeforePreparation)
    {
        ASSERT_TRUE(has_device_) << "Native Q8 metadata preflight requires its device";
        const auto &facts = rocmHardwareMetadata().at(0);
        const auto actual = nativeQuantisedHardwareForDevice(0);
        EXPECT_NO_THROW(actual.requireObserved());
        EXPECT_EQ(actual.ordinal, facts.ordinal);
        EXPECT_EQ(actual.architecture,
            native_quantised::NativeQuantisedArchitecture::fromROCmISATarget(facts.isa_target));
        EXPECT_EQ(device_name_, facts.name + " (" + facts.isa_target + ")");
    }

    /** @brief Every source format proves its actual producer and captured bytes. */
    TEST_F(NativeQuantisedSweepTest, CapturedProducerIdentityAllFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        ASSERT_TRUE(has_device_) << "ROCm producer preflight requires its device";
        const GEMMShape shape{"producer-identity", "square", 256, 256};
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            SCOPED_TRACE(format.name);
            auto weights = format.create(shape.N, shape.K);
            const auto variants = trainerVariantsForFormat(format.name);
            PreparedROCmSweepExecution execution(shape, 64, weights.get(), variants, 1);
            for (const auto &variant : variants)
            {
                SCOPED_TRACE(variant.name);
                clearVNNIPrefillLaunch();
                ASSERT_FALSE(lastVNNIPrefillLaunch());
                const auto result = execution.verifyCaptured(variant);
                EXPECT_EQ(result.byte_mismatches_vs_auto, 0u);
                EXPECT_EQ(result.local_memory_bytes_per_thread, 0u);
                EXPECT_GT(result.registers_per_thread, 0);
                EXPECT_GT(result.max_active_blocks_per_sm, 0);
                if (variant.producer)
                {
                    EXPECT_EQ(result.observed_producer, *variant.producer);
                    EXPECT_EQ(result.observed_n_tile, variant.force_n);
                    EXPECT_EQ(result.observed_m_tile, variant.mt);
                    EXPECT_EQ(result.observed_min_blocks, variant.min_blocks);
                    EXPECT_EQ(result.observed_unroll, variant.unroll < 0 ? 4 : variant.unroll);
                    EXPECT_EQ(result.observed_full_tiles, variant.full_tiles == 1);
                }
            }
        }
#endif
    }

    /** @test PerfStats names the physical producer bound into each retained graph. */
    TEST_F(NativeQuantisedSweepTest, CapturedPrefillPerfStatsAuthenticatesPhysicalProducer)
    {
        ASSERT_TRUE(has_device_);
        const int device_ordinal = wmma_device_ordinal_.value_or(0);
        if (wmma_device_ordinal_) recordWmmaDevice();
        /** @brief Restore the caller's observation policy even after an assertion. */
        struct ObservationScope
        {
            std::array<const char *, 2> names{"LLAMINAR_PERF_STATS_SUMMARY", "LLAMINAR_PERF_STATS_FILTER"};
            std::array<std::optional<std::string>, 2> previous;
            /** @brief Enable only the capture-time producer family at a quiescent test boundary. */
            ObservationScope()
            {
                for (size_t index = 0; index < names.size(); ++index)
                    if (const char *value = std::getenv(names[index])) previous[index] = value;
                setenv(names[0], "1", 1);
                setenv(names[1], "kernel.rocm_native_quantised_prefill_launch", 1);
                PerfStatsCollector::reset();
            }
            /** @brief Restore original filters and retire test observations. */
            ~ObservationScope()
            {
                for (size_t index = 0; index < names.size(); ++index)
                    if (previous[index]) setenv(names[index], previous[index]->c_str(), 1);
                    else unsetenv(names[index]);
                PerfStatsCollector::reset();
            }
        } observation;
        // Native replays do not manufacture additional host launch counts.
        ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
        const GEMMShape shape{"producer_stats", "regression", 65, 256};
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            SCOPED_TRACE(format.name);
            auto weights = format.create(shape.N, shape.K);
            const auto physical = canonicalDeviceVnniCodebookId(requireNativeQuantisedInfo(weights.get(), format.name));
            const auto family = DensePrefillWeightFamily::NativeQuantised;
            const auto inventory = trainerVariantsForFormat(format.name, family, device_ordinal);
            std::vector<VariantConfig> variants;
            const auto cooperative = std::find_if(inventory.begin(), inventory.end(), [](const auto &variant) {
                return variant.producer == VNNIPrefillProducer::NativeCooperative;
            });
            ASSERT_NE(cooperative, inventory.end());
            variants.push_back(*cooperative);
            for (const auto &variant : inventory)
                if (variant.producer == VNNIPrefillProducer::NativeWmma &&
                    (variant.force_n == 32 || variant.force_n == 128)) variants.push_back(variant);
            const bool wmma_hardware = nativeQuantisedHardwareForDevice(device_ordinal).architecture ==
                native_quantised::NativeQuantisedArchitecture::rocm(11, 0, 0);
            ASSERT_EQ(variants.size(), wmma_hardware ? 3u : 1u);
            PreparedROCmSweepExecution execution(shape, 33, weights.get(), variants, 1,
                SweepDispatch::GenericTournament, family, device_ordinal);
            for (const auto &variant : variants)
            {
                SCOPED_TRACE(variant.name);
                PerfStatsCollector::reset();
                const auto result = execution.verifyCaptured(variant);
                ASSERT_TRUE(result.correctness_pass);
                ASSERT_EQ(result.local_memory_bytes_per_thread, 0u);
                ASSERT_EQ(result.observed_producer, *variant.producer);
                const auto launch = lastVNNIPrefillLaunch();
                ASSERT_TRUE(launch);
                const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_quantised_prefill_launch"});
                ASSERT_EQ(records.size(), 1u);
                const auto &record = records.front();
                EXPECT_EQ(record.kind, PerfStatRecord::Kind::Counter);
                EXPECT_EQ(record.domain, "kernel");
                EXPECT_EQ(record.phase, "gemm");
                EXPECT_EQ(record.device, "rocm:" + std::to_string(device_ordinal));
                EXPECT_EQ(record.count, 2u);
                EXPECT_EQ(record.value, 2.0);
                const PerfStatsCollector::Tags expected{
                    {"codebook", std::to_string(physical)},
                    {"arithmetic_policy_codebook", std::to_string(physical == 24 ? 4 : physical)},
                    {"m", "33"}, {"n", "65"}, {"k", "256"},
                    {"producer", vnniPrefillProducerName(*variant.producer)},
                    {"n_tile", std::to_string(variant.force_n)}, {"m_tile", std::to_string(variant.mt)},
                    {"min_blocks", std::to_string(variant.min_blocks)},
                    {"unroll", std::to_string(variant.unroll < 0 ? 4 : variant.unroll)},
                    {"full_tiles", "0"}, {"block_threads", std::to_string(launch->block_threads)},
                    {"dynamic_shared_memory_bytes", std::to_string(launch->dynamic_shared_memory_bytes)}};
                EXPECT_EQ(record.tags, expected);
            }
        }
    }

    /** @test Real retained projections reject omitted quantizers and output producers. */
    TEST_F(NativeQuantisedSweepTest, DirtyCapturedReplayRejectsMissingWork)
    {
        ASSERT_TRUE(has_device_);
        const GEMMShape shape{"dirty_replay", "regression", 512, 2048};
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            SCOPED_TRACE(format.name);
            auto weights = format.create(shape.N, shape.K);
            const auto variants = trainerVariantsForFormat(format.name);
            const auto auto_variant = std::find_if(variants.begin(), variants.end(),
                [](const auto &variant) { return variant.name == "Auto"; });
            ASSERT_NE(auto_variant, variants.end());
            PreparedROCmSweepExecution execution(shape, 64, weights.get(), {*auto_variant}, 1,
                SweepDispatch::GenericTournament, DensePrefillWeightFamily::NativeQuantised);
            execution.verifyDirtyReplayNegativeControls(*auto_variant);
        }
    }

    /** @brief Replay every source decoder through native WMMA, including both output tails. */
    TEST_F(NativeQuantisedSweepTest, WmmaCapturedReplayAllFormats)
    {
        ASSERT_TRUE(has_device_);
        if (!wmma_device_ordinal_)
            GTEST_SKIP() << "Native WMMA replay requires gfx1100";
        const int device_ordinal = *wmma_device_ordinal_;
        recordWmmaDevice();
        ScopedROCmNativeQuantisedTrainerOverride partition_guard;
        const int original_partitions = debugEnv().rocm.native_quantised_gemv_kb;
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            SCOPED_TRACE(format.name);
            const auto family = std::optional{DensePrefillWeightFamily::NativeQuantised};
            auto variants = trainerVariantsForFormat(format.name, family, device_ordinal);
            std::erase_if(variants, [](const auto &variant) {
                return variant.producer != VNNIPrefillProducer::NativeWmma;
            });
            ASSERT_EQ(variants.size(), 4u);
            std::vector<std::tuple<int, int, int>> geometries{
                {32, 64, 256}, {33, 65, 512}, {65, 65, 1280}, {64, 96, 768}, {127, 63, 256}, {65, 129, 256}};
            // Exhaust the small/grouped row surface rather than sampling only
            // tile boundaries, then exercise every larger prefill anchor.
            for (int m = 1; m <= 129; ++m) geometries.emplace_back(m, 65, 256);
            for (const int m : {256, 512, 1024}) geometries.emplace_back(m, 129, 256);
            for (const int n : {1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128})
                geometries.emplace_back(33, n, 256);
            const auto source = std::find_if(native_quantised_formats::kAllSourceFormats.begin(),
                native_quantised_formats::kAllSourceFormats.end(),
                [&](const auto &candidate) { return candidate.quant_type == format.name; });
            ASSERT_NE(source, native_quantised_formats::kAllSourceFormats.end());
            if (!source->metadata->is_superblock)
                for (const int k : {32, 96, 160}) geometries.emplace_back(33, 65, k);
            for (const auto [m, n, k] : geometries)
            {
                mutableDebugEnv().rocm.native_quantised_gemv_kb =
                    k == 768 ? 5 : k == 1280 ? 3 : original_partitions;
                SCOPED_TRACE(::testing::Message() << "M=" << m << " N=" << n << " K=" << k);
                const GEMMShape shape{"wmma-replay", "tail", n, k};
                auto weights = format.create(shape.N, shape.K);
                PreparedROCmSweepExecution execution(shape, m, weights.get(), variants, 1,
                    SweepDispatch::GenericTournament, family, device_ordinal);
                for (const auto &variant : variants)
                {
                    SCOPED_TRACE(variant.name);
                    // Ordinary M1 routes through GEMV. Prove the WMMA leaf
                    // explicitly against that complete serial oracle; larger
                    // rows replay the complete production projection pipeline.
                    const auto result = execution.verifyCaptured(variant, m == 1
                        ? SweepSubmission::CapturedProducer : SweepSubmission::Captured);
                    EXPECT_EQ(result.observed_producer, VNNIPrefillProducer::NativeWmma);
                    EXPECT_EQ(result.byte_mismatches_vs_auto, 0u);
                    EXPECT_EQ(result.local_memory_bytes_per_thread, 0u);
                    EXPECT_GT(result.max_active_blocks_per_sm, 0);
                }
            }
        }
    }

    /** @brief Prequantized replay preserves signed zero, K tails and partition boundaries. */
    TEST_F(NativeQuantisedSweepTest, WmmaWholeRowScaleAndPartialKReplay)
    {
        ASSERT_TRUE(has_device_);
        if (!wmma_device_ordinal_)
            GTEST_SKIP() << "Native WMMA replay requires gfx1100";
        const int device_ordinal = *wmma_device_ordinal_;
        recordWmmaDevice();
        ScopedROCmNativeQuantisedTrainerOverride override;
        ScopedDensePrefillLookup lookup(SweepDispatch::GenericTournament);
        const auto check = [](hipError_t error) {
            if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
        };
        check(hipSetDevice(device_ordinal));
        ScopedGPUStream stream_owner(DeviceId::rocm(device_ordinal));
        const auto stream = static_cast<hipStream_t>(stream_owner.get());
        constexpr int m = 33, n = 65;
        auto variants = trainerVariantsForFormat("Q4_0", DensePrefillWeightFamily::NativeQuantised, device_ordinal);
        std::erase_if(variants, [](const auto &variant) {
            return variant.producer != VNNIPrefillProducer::NativeWmma;
        });
        for (const uint8_t codebook : {uint8_t{0}, kNativeQuantisedExpandedInt8MinCodebook})
        for (const int k : {32, 96, 160, 256})
        for (const int partitions : {1, 3, 5})
        for (const float row_scale : {1.0f, 0x1p-126f})
        {
            SCOPED_TRACE(::testing::Message() << "codebook=" << int(codebook) << " K=" << k << " partitions=" << partitions
                << " row_scale=" << row_scale);
            mutableDebugEnv().rocm.native_quantised_gemv_kb = partitions;
            const int blocks = k / 32;
            // These tensor-owned byte banks retain admission/allocation ownership.
            // The test intentionally supplies prequantized source bytes so the
            // whole-row scale ABI is exercised without activation requantization.
            auto a = TestTensorFactory::createFP32Zeros({size_t(m * k / 4)});
            const int payload_bytes = codebook == 0 ? 16 : 32;
            auto payload = TestTensorFactory::createFP32({size_t(blocks * n * payload_bytes / 4)});
            auto scales = TestTensorFactory::createFP32Zeros({size_t((blocks * n + 1) / 2)});
            auto minima = TestTensorFactory::createFP32Zeros({size_t((blocks * n + 1) / 2)});
            auto rows = TestTensorFactory::createFP32({m});
            auto output = TestTensorFactory::createFP32({m, n});
            auto oracle = TestTensorFactory::createFP32({m, n});
            auto *ab = reinterpret_cast<int8_t *>(a->mutable_data());
            auto *pb = reinterpret_cast<uint8_t *>(payload->mutable_data());
            std::fill_n(pb, blocks * n * payload_bytes, codebook == 0 ? uint8_t{0x88} : uint8_t{0});
            const std::vector<uint16_t> scale_bits(blocks * n, uint16_t{1}); // exact half 2^-24
            std::memcpy(scales->mutable_data(), scale_bits.data(), scale_bits.size() * sizeof(uint16_t));
            std::fill_n(rows->mutable_data(), m, row_scale);
            for (int row = 0; row < m; ++row) ab[row * k] = 1;
            for (int column = 0; column < n; ++column)
                pb[column * payload_bytes] = codebook == 0 ? 0x87 : 0xff;
            for (auto *tensor : {a.get(), payload.get(), scales.get(), minima.get(), rows.get()})
                ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::rocm(device_ordinal), stream));
            ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(device_ordinal), stream));
            ASSERT_TRUE(oracle->allocateOnDevice(DeviceId::rocm(device_ordinal), stream));
            const auto launch = [&](FP32Tensor *destination) {
                return rocmGemm_native_quantised_fp32_with_policy(
                    static_cast<const int8_t *>(a->gpu_data_ptr()),
                    static_cast<const uint8_t *>(payload->gpu_data_ptr()), scales->gpu_data_ptr(),
                    codebook == kNativeQuantisedExpandedInt8MinCodebook ? minima->gpu_data_ptr() : nullptr, nullptr,
                    static_cast<float *>(destination->gpu_data_ptr()),
                    static_cast<const float *>(rows->gpu_data_ptr()), nullptr, m, n, k,
                    codebook, codebook == kNativeQuantisedExpandedInt8MinCodebook ? 5 : 0, device_ordinal, stream);
            };
            override.apply(VariantConfig{"Auto", -1, -1, -1, -1, -1, std::nullopt});
            ASSERT_TRUE(launch(oracle.get()));
            std::vector<uint32_t> reference(m * n), observed(m * n);
            check(hipMemcpyAsync(reference.data(), oracle->gpu_data_ptr(), reference.size() * sizeof(uint32_t),
                hipMemcpyDeviceToHost, stream));
            check(hipStreamSynchronize(stream));
            const uint32_t expected = row_scale == 1.0f ? std::bit_cast<uint32_t>(-0x1p-24f) :
                std::min(partitions, blocks) == 1 ? 0x80000000u : 0u;
            ASSERT_TRUE(std::all_of(reference.begin(), reference.end(),
                [&](uint32_t bits) { return bits == expected; }));
            for (const auto &variant : variants)
            {
                SCOPED_TRACE(variant.name);
                override.apply(variant);
                HIPGraphCapture graph(stream, device_ordinal);
                ScopedBackendGraphCapture capture(graph, "WMMA whole-row arithmetic replay");
                ASSERT_TRUE(capture.begin());
                ASSERT_TRUE(launch(output.get()));
                capture.finish();
                ASSERT_TRUE(graph.instantiate());
                for (const int poison : {0xff, 0x55, 0})
                {
                    check(hipMemsetAsync(output->gpu_data_ptr(), poison, observed.size() * sizeof(uint32_t), stream));
                    ASSERT_TRUE(graph.launch());
                    check(hipMemcpyAsync(observed.data(), output->gpu_data_ptr(), observed.size() * sizeof(uint32_t),
                        hipMemcpyDeviceToHost, stream));
                    check(hipStreamSynchronize(stream));
                    EXPECT_EQ(observed, reference);
                }
            }
        }
    }

    // =========================================================================
    // Test: Full sweep — all variants × all shapes × all M values
    // =========================================================================

    TEST_F(NativeQuantisedSweepTest, Q4_0_VariantSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeQuantised Q4_0 Sweep] Variant Sweep\n");
        fprintf(stderr, "[NativeQuantised Q4_0 Sweep] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeQuantised Q4_0 Sweep] %zu shapes × %zu M values × %zu variants\n",
                GEMM_SHAPES.size(), M_VALUES.size(), NATIVE_QUANTISED_VARIANTS.size());
        fprintf(stderr, "[NativeQuantised Q4_0 Sweep] %d warmup + %d timed runs each\n",
                WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr,
                "[NativeQuantised Q4_0 Sweep] Correctness gate: full output byte equality vs Auto\n\n");

        const size_t num_shapes = GEMM_SHAPES.size();
        const size_t num_m = M_VALUES.size();
        const size_t num_variants = NATIVE_QUANTISED_VARIANTS.size();

        std::vector<BenchResult> results(num_shapes * num_m * num_variants);

        int total_benchmarks = static_cast<int>(num_shapes * num_m * num_variants);
        int completed = 0;

        for (size_t si = 0; si < num_shapes; ++si)
        {
            const auto &shape = GEMM_SHAPES[si];

            // Create Q4_0 weights once per shape (shared across all variants + M values)
            auto q4_weights = TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});

            for (size_t mi = 0; mi < num_m; ++mi)
            {
                int M = M_VALUES[mi];
                PreparedROCmSweepExecution execution(
                    shape,
                    M,
                    q4_weights.get(),
                    NATIVE_QUANTISED_VARIANTS,
                    BENCH_RUNS);

                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &variant = NATIVE_QUANTISED_VARIANTS[vi];
                    auto r = execution.measure(
                        variant,
                        WARMUP_RUNS,
                        BENCH_RUNS);
                    results[si * num_m * num_variants + mi * num_variants + vi] = std::move(r);
                    ++completed;

                    const auto &res = results[si * num_m * num_variants + mi * num_variants + vi];
                    EXPECT_EQ(res.byte_mismatches_vs_auto, 0u)
                        << shape.name << " M=" << M
                        << " candidate=" << variant.name
                        << " first_byte_mismatch="
                        << res.first_byte_mismatch_vs_auto;
                    fprintf(stderr, "  %s %s M=%d: %.0f μs (byte_mismatches=%zu) [%d/%d]\n",
                            variant.name.c_str(), shape.name.c_str(), M,
                            res.min_us, res.byte_mismatches_vs_auto,
                            completed, total_benchmarks);
                }
            }
        }

        // =====================================================================
        // Render per-M comparison tables
        // =====================================================================
        for (size_t mi = 0; mi < num_m; ++mi)
        {
            int M = M_VALUES[mi];

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Header
            table << fort::header << "Shape" << "Cat" << "N" << "K";
            for (const auto &v : NATIVE_QUANTISED_VARIANTS)
                table << v.name;
            table << "Best" << "vs Auto" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::left);

            for (size_t si = 0; si < num_shapes; ++si)
            {
                const auto &shape = GEMM_SHAPES[si];
                table << shape.name << shape.category
                      << std::to_string(shape.N) << std::to_string(shape.K);

                double best_us = 1e18;
                int best_idx = -1;
                double auto_us = 0.0;

                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &res = results[si * num_m * num_variants + mi * num_variants + vi];
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.0f", res.min_us);
                    table << buf;

                    if (res.min_us > 0 && res.min_us < best_us && res.correctness_pass)
                    {
                        best_us = res.min_us;
                        best_idx = static_cast<int>(vi);
                    }

                    if (NATIVE_QUANTISED_VARIANTS[vi].name == "Auto")
                        auto_us = res.min_us;
                }

                // Best variant name
                if (best_idx >= 0)
                    table << NATIVE_QUANTISED_VARIANTS[static_cast<size_t>(best_idx)].name;
                else
                    table << "N/A";

                // Speedup vs Auto
                if (auto_us > 0 && best_us > 0 && best_us < 1e17)
                {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.2fx", auto_us / best_us);
                    table << buf;
                }
                else
                    table << "N/A";

                table << fort::endr;
            }

            fprintf(stderr, "\n=== M=%d Comparison ===\n%s\n", M, table.to_string().c_str());
        }

        // =====================================================================
        // Render per-shape×M best variant summary (CSV-friendly)
        // =====================================================================
        fprintf(stderr, "\n=== BEST VARIANT SUMMARY (CSV) ===\n");
        fprintf(stderr, "Shape,M,N,K,Best_Variant,Best_us,Auto_us,Speedup\n");

        for (size_t si = 0; si < num_shapes; ++si)
        {
            const auto &shape = GEMM_SHAPES[si];
            for (size_t mi = 0; mi < num_m; ++mi)
            {
                int M = M_VALUES[mi];
                double best_us = 1e18;
                int best_idx = -1;
                double auto_us = 0.0;

                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &res = results[si * num_m * num_variants + mi * num_variants + vi];
                    if (res.min_us > 0 && res.min_us < best_us && res.correctness_pass)
                    {
                        best_us = res.min_us;
                        best_idx = static_cast<int>(vi);
                    }
                    if (NATIVE_QUANTISED_VARIANTS[vi].name == "Auto")
                        auto_us = res.min_us;
                }

                const char *best_name = (best_idx >= 0) ? NATIVE_QUANTISED_VARIANTS[static_cast<size_t>(best_idx)].name.c_str() : "N/A";
                double speedup = (auto_us > 0 && best_us > 0 && best_us < 1e17)
                                     ? (auto_us / best_us)
                                     : 0.0;

                fprintf(stderr, "%s,%d,%d,%d,%s,%.1f,%.1f,%.2f\n",
                        shape.name.c_str(), M, shape.N, shape.K,
                        best_name, best_us, auto_us, speedup);
            }
        }
#endif
    }

    /**
     * @brief Measure installed Auto at canonical projection shapes in a graph.
     *
     * This is deliberately not a dispatch-training transaction. Geometry and
     * formats come from the same inventories as the trainer, every sample is
     * retained, and the full output is byte-compared with the exact-M oracle.
     * Callers map these physical N/K shapes to their model's TP roles; a smaller
     * local matrix is not proof that the surrounding multi-device graph works.
     * Producer-only timing covers NativeQuantised matrices, not the distinct INT8
     * producer selected by production for Q8 weights. Such requests fail
     * explicitly; the probe must never substitute another weight format.
     */
    TEST_F(NativeQuantisedSweepTest, CapturedProductionShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";
        auto formats = getEnvCsvSet("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_FORMATS");
        if (formats.empty()) formats.insert("q6_k");
        auto shapes = getEnvCsvSet("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_SHAPES");
        if (shapes.empty()) shapes.insert("3b_attnout");
        const auto requested_operation = getEnvString("LLAMINAR_ROCM_NATIVE_QUANTISED_PROBE_OPERATION");
        const std::string operation = requested_operation.empty() ? "pipeline" : requested_operation;
        ASSERT_TRUE(operation == "pipeline" || operation == "producer")
            << "Probe operation must be pipeline or producer";
        const auto submission = operation == "producer"
            ? SweepSubmission::CapturedProducer : SweepSubmission::Captured;
        const auto rows = getEnvCsvInts("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_M", {64, 512});
        const int warmups = std::max(1, getEnvInt("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_WARMUP").value_or(10));
        const int repetitions = std::max(1, getEnvInt("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_BENCH").value_or(31));
        const auto selected = std::find_if(NATIVE_QUANTISED_VARIANTS.begin(), NATIVE_QUANTISED_VARIANTS.end(),
            [](const VariantConfig &variant) { return variant.name == "Auto"; });
        ASSERT_NE(selected, NATIVE_QUANTISED_VARIANTS.end());
        int cases = 0;
        std::printf("format,shape,m,n,k,sample_index,latency_us,byte_mismatches,registers,scratch_bytes,n_tile,m_tile,operations_per_sample,operation\n");
        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            if (!shouldRunName(formats, format.name)) continue;
            SCOPED_TRACE(format.name);
            for (const auto &shape : GEMM_SHAPES)
            {
                if (!shouldRunName(shapes, shape.name)) continue;
                SCOPED_TRACE(shape.name);
                auto weights = format.create(shape.N, shape.K);
                for (const int m : rows)
                {
                    PreparedROCmSweepExecution execution(shape, m, weights.get(), {*selected}, repetitions,
                        SweepDispatch::InstalledProduction);
                    const auto result = execution.measure(*selected, warmups, repetitions, submission);
                    ASSERT_EQ(result.byte_mismatches_vs_auto, 0u) << shape.name << " M=" << m;
                    const auto samples = execution.timingSamples();
                    for (size_t index = 0; index < samples.size(); ++index)
                        std::printf("%s,%s,%d,%d,%d,%zu,%.9f,%zu,%d,%zu,%d,%d,16,%s\n",
                            format.name.c_str(), shape.name.c_str(), m, shape.N, shape.K,
                            index, samples[index], result.byte_mismatches_vs_auto,
                            result.registers_per_thread, result.local_memory_bytes_per_thread,
                            result.observed_n_tile, result.observed_m_tile, operation.c_str());
                    ++cases;
                }
            }
        }
        ASSERT_GT(cases, 0) << "No canonical production geometry matched the requested filters";
#endif
    }

    /**
     * @brief Prove every measured producer through real captured replay.
     *
     * Physical keys are read from the immutable table. Source-format fixtures
     * retain all 21 registries, including aliases with different source math.
     * The exact-M oracle deliberately uses generic Auto, so selecting the
     * installed table cannot certify itself. Resource identity, poisoned output
     * and complete byte equality are required for every captured candidate.
     */
    /**
     * @brief Twelve disjoint source-format groups bound fixture preprocessing cost.
     * The complete generated geometry inventory is retained in every group.
     */
    class InstalledDensePrefillOverlayTest : public NativeQuantisedSweepTest,
                                            public ::testing::WithParamInterface<int> {};

    /** @brief Keep every source format while bounding each registered setup transaction. */
    inline constexpr int kInstalledDensePrefillSourceGroups = 12;

    TEST_P(InstalledDensePrefillOverlayTest, AllSourceFormats)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        ASSERT_TRUE(has_device_);
        const auto actual_hardware = nativeQuantisedHardwareForDevice(0);
        const auto installed_entries = densePrefillInstalledEntries();
        std::map<std::pair<int, int>, std::set<int>> geometries;
        std::size_t geometry_count = 0;
        for (const auto &entry : installed_entries)
            if (!entry.architecture.observed() &&
                geometries[{entry.n, entry.k}].insert(entry.m).second) ++geometry_count;
        ASSERT_FALSE(geometries.empty());
        ASSERT_GE(GetParam(), 0);
        ASSERT_LT(GetParam(), kInstalledDensePrefillSourceGroups);
        const auto selected = std::find_if(NATIVE_QUANTISED_VARIANTS.begin(), NATIVE_QUANTISED_VARIANTS.end(),
            [](const VariantConfig &variant) { return variant.name == "Auto"; });
        ASSERT_NE(selected, NATIVE_QUANTISED_VARIANTS.end());
        std::size_t cases = 0;
        for (std::size_t format_index = static_cast<std::size_t>(GetParam());
             format_index < NATIVE_QUANTISED_FORMATS.size(); format_index += kInstalledDensePrefillSourceGroups)
        {
            const auto &format = NATIVE_QUANTISED_FORMATS[format_index];
            SCOPED_TRACE(format.name);
            for (const auto &[projection, row_buckets] : geometries)
            {
                const auto [n, k] = projection;
                const GEMMShape shape{"installed_dense_prefill", "measured", n, k};
                // Source weights do not depend on M. Reuse their expensive
                // random construction while each row bucket still prepares
                // its own arena, independent oracle and captured producer.
                auto weights = format.create(n, k);
                const auto codebook = canonicalDeviceVnniCodebookId(
                    requireNativeQuantisedInfo(weights.get(), "installed overlay source weights"));
                // This canonical fixture uses the ordinary packer: Q8 is
                // expanded INT8, while low-bit formats retain native payloads.
                // Native-Q8 is a separate physical contract covered by the
                // focused prepared-family regression, not a measured row here.
                const auto family = codebook == 19
                    ? DensePrefillWeightFamily::Int8Blockwise : DensePrefillWeightFamily::NativeQuantised;
                for (const int m : row_buckets)
                {
                    SCOPED_TRACE(::testing::Message() << "M=" << m << " N=" << n << " K=" << k);
                    auto exact = selectDensePrefillExactConfig(codebook, m, n, k, family, &actual_hardware);
                    if (!exact && codebook == kNativeQuantisedIQ4XSSourceScaleCodebook)
                        exact = selectDensePrefillExactConfig(4, m, n, k, family);
                    ASSERT_TRUE(exact);
                    PreparedROCmSweepExecution execution(shape, m, weights.get(), {*selected}, 1,
                        SweepDispatch::InstalledProduction);
                    const auto result = execution.verifyCaptured(*selected);
                    EXPECT_EQ(result.observed_producer, exact->producer);
                    EXPECT_EQ(result.observed_n_tile, exact->n_tile);
                    EXPECT_EQ(result.observed_m_tile, exact->m_tile);
                    EXPECT_EQ(result.observed_min_blocks, exact->min_blocks);
                    EXPECT_EQ(result.observed_unroll, exact->unroll);
                    EXPECT_EQ(result.observed_full_tiles, exact->full_tiles);
                    EXPECT_EQ(result.local_memory_bytes_per_thread, 0u);
                    EXPECT_GT(result.max_active_blocks_per_sm, 0);
                    ASSERT_EQ(result.byte_mismatches_vs_auto, 0u);
                    ++cases;
                }
            }
        }
        std::size_t unavailable_native_cases = 0;
        std::size_t native_cases = 0;
        std::size_t native_wmma_cases = 0;
        std::size_t expected_native_wmma_cases = 0;
        std::set<std::array<uint32_t, 4>> tested_architectures;
        // Qualified rows must reach every visible native ISA, even when the
        // first card belongs to a different cohort. Equivalent cards share
        // this functional obligation; no visibility mask or inventory changes.
        for (size_t ordinal = 0; ordinal < rocmHardwareMetadata().size(); ++ordinal)
        {
            const auto native_hardware = nativeQuantisedHardwareForDevice(static_cast<int>(ordinal));
            if (!tested_architectures.insert(native_hardware.architecture.scalars()).second) continue;
            SCOPED_TRACE(::testing::Message() << "device=" << ordinal);
            for (std::size_t format_index = static_cast<std::size_t>(GetParam());
                 format_index < NATIVE_QUANTISED_FORMATS.size(); format_index += kInstalledDensePrefillSourceGroups)
            {
                const auto &format = NATIVE_QUANTISED_FORMATS[format_index];
                const auto source = std::find_if(native_quantised_formats::kAllSourceFormats.begin(),
                    native_quantised_formats::kAllSourceFormats.end(),
                    [&](const auto &candidate) { return candidate.quant_type == format.name; });
                ASSERT_NE(source, native_quantised_formats::kAllSourceFormats.end());
                const auto codebook = canonicalDeviceVnniCodebookId(*source->metadata);
                // Derive this obligation directly from installed exact-ISA
                // rows. Lookup and execution agreeing on a retained legacy
                // producer must not conceal a missed WMMA installation.
                expected_native_wmma_cases += std::count_if(
                    std::begin(installed_entries),
                    std::end(installed_entries), [&](const auto &row) {
                        return row.codebook == codebook &&
                            row.family == DensePrefillWeightFamily::NativeQuantised &&
                            row.architecture == native_hardware.architecture &&
                            row.config.producer == VNNIPrefillProducer::NativeWmma;
                    });
                std::set<std::tuple<int, int, int>> tested;
                for (const auto &entry : installed_entries)
                {
                    if (entry.codebook != codebook || entry.family != DensePrefillWeightFamily::NativeQuantised ||
                        !entry.architecture.observed() || !tested.emplace(entry.m, entry.n, entry.k).second) continue;
                    const auto exact = selectDensePrefillExactConfig(codebook, entry.m, entry.n, entry.k,
                        DensePrefillWeightFamily::NativeQuantised, &native_hardware);
                    if (!exact)
                    {
                        // A native-ISA-only entry cannot execute on another card;
                        // retain the unsupported count rather than certify a substitute.
                        ++unavailable_native_cases;
                        continue;
                    }
                    auto weights = format.create(entry.n, entry.k);
                    const GEMMShape shape{"installed_native_prepared", "measured", entry.n, entry.k};
                    PreparedROCmSweepExecution execution(shape, entry.m, weights.get(), {*selected}, 1,
                        SweepDispatch::InstalledProduction, DensePrefillWeightFamily::NativeQuantised,
                        static_cast<int>(ordinal));
                    const auto result = execution.verifyCaptured(*selected);
                    EXPECT_EQ(result.observed_producer, exact->producer);
                    EXPECT_EQ(result.observed_n_tile, exact->n_tile);
                    EXPECT_EQ(result.observed_m_tile, exact->m_tile);
                    EXPECT_EQ(result.observed_min_blocks, exact->min_blocks);
                    EXPECT_EQ(result.observed_unroll, exact->unroll);
                    EXPECT_EQ(result.observed_full_tiles, exact->full_tiles);
                    EXPECT_EQ(result.local_memory_bytes_per_thread, 0u);
                    EXPECT_GT(result.max_active_blocks_per_sm, 0);
                    ASSERT_EQ(result.byte_mismatches_vs_auto, 0u);
                    ++native_cases;
                    if (result.observed_producer == VNNIPrefillProducer::NativeWmma) ++native_wmma_cases;
                }
            }
        }
        // The existing source-group gate exercises each independently prepared
        // alias. Absence of new native rows leaves the original gate unchanged.
        RecordProperty("installed_native_prepared_cases", native_cases);
        RecordProperty("installed_native_wmma_cases", native_wmma_cases);
        RecordProperty("expected_native_wmma_cases", expected_native_wmma_cases);
        EXPECT_EQ(native_wmma_cases, expected_native_wmma_cases);
        RecordProperty("unavailable_native_prepared_cases", unavailable_native_cases);
        const std::size_t source_count =
            (NATIVE_QUANTISED_FORMATS.size() + kInstalledDensePrefillSourceGroups - 1u -
                static_cast<std::size_t>(GetParam())) / kInstalledDensePrefillSourceGroups;
        EXPECT_EQ(cases, source_count * geometry_count);
#endif
    }

    INSTANTIATE_TEST_SUITE_P(InstalledDensePrefill, InstalledDensePrefillOverlayTest,
                             ::testing::Range(0, kInstalledDensePrefillSourceGroups));

    /** @test Authenticate the linked selector's complete table without opening a GPU.
     * Canonical decimal rows retain every key/config/architecture field and end
     * in a newline. Python independently parses the frozen generated include;
     * matching hashes join those bytes to the measured executable/core closure.
     * This cold query never supplies timing or recertifies retained observations.
     */
    TEST(ROCmDensePrefillPolicy, CompiledInventory)
    {
#ifdef HAVE_ROCM
        const auto entries = densePrefillInstalledEntries();
        ASSERT_FALSE(entries.empty());
        std::string canonical = "rocm-dense-prefill-installed-rows-v1\n";
        for (const auto &entry : entries)
        {
            const auto &config = entry.config;
            const std::array<std::int64_t,15> fields{entry.codebook,entry.m,entry.n,entry.k,
                static_cast<int>(entry.family),config.n_tile,config.m_tile,config.min_blocks,
                config.unroll,int(config.full_tiles),static_cast<int>(config.producer),
                static_cast<int>(entry.architecture.backend),entry.architecture.major,
                entry.architecture.minor,entry.architecture.stepping};
            for (std::size_t index = 0; index < fields.size(); ++index)
            {
                if (index) canonical += ',';
                canonical += std::to_string(fields[index]);
            }
            canonical += '\n';
        }
        std::string error;
        const auto digest = sha256BytesHex(canonical,&error);
        ASSERT_TRUE(digest.has_value()) << error;
        std::cout << "ROCM_DENSE_PREFILL_POLICY_V1," << entries.size() << ",sha256:" << *digest << '\n';
#else
        GTEST_SKIP() << "HAVE_ROCM not defined";
#endif
    }

    TEST_F(NativeQuantisedSweepTest, TrainerCsv_CodebookTagged)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const auto preparation = getEnvString("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_PREPARED_FAMILY");
        ASSERT_TRUE(preparation.empty() || preparation == "native_quantised")
            << "Unknown explicit ROCm sweep prepared family";
        const bool native_prepared = !preparation.empty();
        const auto prepared_family = native_prepared
            ? std::optional{DensePrefillWeightFamily::NativeQuantised} : std::nullopt;
        // These are immutable discovery facts, not a new runtime query or a
        // marketing-name target. The child sees one HIP-selected ordinal zero.
        std::string native_csv_suffix;
        if (native_prepared)
        {
            const auto &facts = rocmHardwareMetadata().at(0);
            const auto actual = nativeQuantisedHardwareForDevice(0);
            const auto target = native_quantised::nativeQuantisedArchitectureName(actual.architecture);
            const auto expected = getEnvString("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_ARCHITECTURE");
            if (expected.empty() || expected != target)
                throw std::invalid_argument("native sweep architecture differs from its immutable plan");
            std::string uuid;
            constexpr char hex[] = "0123456789abcdef";
            for (const auto byte : facts.uuid)
            {
                uuid += hex[byte >> 4];
                uuid += hex[byte & 15];
            }
            const auto expected_uuid = getEnvString("LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_EXPECTED_UUID");
            if (!expected_uuid.empty() && expected_uuid != uuid)
                throw std::invalid_argument("native prefill profiler selected a different physical participant");
            native_csv_suffix = ",native_quantised," + target + "," + uuid + "," +
                facts.pci_address + "," + std::to_string(facts.ordinal);
        }
        std::set<std::string> format_filters = getEnvCsvSet("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_FORMATS");
        if (format_filters.empty())
            format_filters.insert(native_prepared ? "q8_0" : "q4_0");
        const std::set<std::string> shape_filters = getEnvCsvSet("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_SHAPES");
        const std::set<std::string> variant_filters = getEnvCsvSet("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_VARIANTS");
        const std::vector<int> m_values = getEnvCsvInts(
            "LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_M",
            supportedPrefillGraphBucketSizes());
        const int warmup_runs = std::max(
            1,
            getEnvInt("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_WARMUP")
                .value_or(WARMUP_RUNS));
        const int bench_runs = std::max(
            1,
            getEnvInt("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_BENCH")
                .value_or(BENCH_RUNS));
        const int max_cases = std::max(1, getEnvInt("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_MAX_CASES").value_or(1));
        const std::string csv_path = getEnvString("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_CSV");
        const std::string timing_csv_path =
            getEnvString("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_TIMING_CSV");

        const auto profiler_request = llaminar2::test::native_quantised_dispatch::profilerRequestId();
        const auto profiler_candidate = llaminar2::test::native_quantised_dispatch::profilerEnvironment(
            "LLAMINAR_ROCM_DENSE_PRODUCTION_PROFILE_CANDIDATE");
        if (profiler_request.empty() != profiler_candidate.empty())
            throw std::invalid_argument("ROCm dense profiling requires both request ID and exact candidate");
        if (!profiler_request.empty())
        {
            if (!native_prepared)
                throw std::invalid_argument("dense profiling requires explicit architecture-qualified native preparation");
            if (format_filters.size() != 1 || shape_filters.size() != 1 || m_values.size() != 1 ||
                max_cases != 1 || getEnvString("LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_FORMATS").empty())
                throw std::invalid_argument("ROCm dense profiling requires one explicit format, shape, M and MAX_CASES=1");
            const auto format = std::find_if(NATIVE_QUANTISED_FORMATS.begin(), NATIVE_QUANTISED_FORMATS.end(),
                [&](const auto &entry) { return shouldRunName(format_filters, entry.name); });
            const auto shape = std::find_if(GEMM_SHAPES.begin(), GEMM_SHAPES.end(),
                [&](const auto &entry) { return shouldRunName(shape_filters, entry.name); });
            if (format == NATIVE_QUANTISED_FORMATS.end() || shape == GEMM_SHAPES.end())
                throw std::invalid_argument("ROCm dense profiler has an unknown source format or shape");
            const auto variants = trainerVariantsForFormat(format->name, prepared_family);
            const auto candidate = std::find_if(variants.begin(), variants.end(),
                [&](const auto &entry) { return entry.name == profiler_candidate; });
            if (candidate == variants.end())
                throw std::invalid_argument("ROCm dense profiler candidate is absent or incompatible with its prepared format");
            auto weights = format->create(static_cast<size_t>(shape->N), static_cast<size_t>(shape->K));
            PreparedROCmSweepExecution execution(*shape, m_values.front(), weights.get(), {*candidate}, 1,
                native_prepared ? SweepDispatch::InstalledProduction : SweepDispatch::GenericTournament, prepared_family);
            execution.profileExactCandidate(*candidate, profiler_request, warmup_runs);
            return;
        }

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open ROCm NativeQuantised sweep CSV: " << csv_path;
            std::fprintf(csv,
                         "backend,phase,format,codebook,execution_codebook,shape,category,m,n,k,variant,min_us,mean_us,stddev_us,gflops,cosine,correctness_pass,byte_mismatches_vs_auto,first_byte_mismatch_vs_auto,is_best,observed_n_tile,observed_m_tile,observed_min_blocks,observed_unroll,observed_full_tiles,registers_per_thread,local_memory_bytes_per_thread,static_shared_memory_bytes,max_threads_per_block,max_active_blocks_per_sm,observed_producer,dynamic_shared_memory_bytes%s\n", native_prepared ? ",prepared_family,gpu_architecture,gpu_uuid,gpu_pci,native_ordinal" : "");
        }

        std::FILE *timing_csv = nullptr;
        if (!timing_csv_path.empty())
        {
            timing_csv = std::fopen(timing_csv_path.c_str(), "w");
            ASSERT_NE(timing_csv, nullptr)
                << "Failed to open ROCm NativeQuantised timing CSV: "
                << timing_csv_path;
            std::fprintf(
                timing_csv,
                "backend,phase,format,codebook,execution_codebook,shape,m,n,k,variant,"
                "sample_index,timed_replays,latency_us,latency_us_hex,captured_operations%s\n", native_prepared ? ",prepared_family,gpu_architecture,gpu_uuid,gpu_pci,native_ordinal" : "");
        }

        int executed_cases = 0;
        int executed_rows = 0;

        for (const auto &format : NATIVE_QUANTISED_FORMATS)
        {
            if (!shouldRunName(format_filters, format.name))
                continue;

            const std::vector<VariantConfig> launchable_variants =
                trainerVariantsForFormat(format.name, prepared_family);
            std::vector<VariantConfig> selected_variants;
            std::copy_if(
                launchable_variants.begin(),
                launchable_variants.end(),
                std::back_inserter(selected_variants),
                [&variant_filters](const VariantConfig &variant)
                { return shouldRunName(variant_filters, variant.name); });
            ASSERT_FALSE(selected_variants.empty())
                << "No ROCm NativeQuantised variants selected for " << format.name;

            for (const auto &shape : GEMM_SHAPES)
            {
                if (!shouldRunName(shape_filters, shape.name))
                    continue;
                if ((shape.K % 32) != 0)
                    continue;

                auto weights = format.create(static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
                const auto &source_format = requireNativeQuantisedInfo(weights.get(), format.name);
                const uint8_t codebook_id = source_format.codebook_id;
                const uint8_t execution_codebook_id = canonicalDeviceVnniCodebookId(source_format);

                for (const int M : m_values)
                {
                    if (executed_cases >= max_cases)
                        break;

                    PreparedROCmSweepExecution execution(
                        shape,
                        M,
                        weights.get(),
                        selected_variants,
                        bench_runs, native_prepared ? SweepDispatch::InstalledProduction : SweepDispatch::GenericTournament, prepared_family);

                    std::vector<BenchResult> rows;
                    rows.reserve(selected_variants.size());
                    for (const auto &variant : selected_variants)
                    {
                        auto r = execution.measure(
                            variant, warmup_runs, bench_runs,
                            SweepSubmission::Captured);
                        EXPECT_EQ(r.byte_mismatches_vs_auto, 0u)
                            << format.name << ' ' << shape.name << " M=" << M
                            << " candidate=" << variant.name
                            << " first_byte_mismatch="
                            << r.first_byte_mismatch_vs_auto;
                        r.correctness_pass =
                            r.correctness_pass &&
                            r.byte_mismatches_vs_auto == 0;
                        if (timing_csv)
                        {
                            const auto samples = execution.timingSamples();
                            for (size_t sample_index = 0;
                                 sample_index < samples.size(); ++sample_index)
                            {
                                const double latency_us = samples[sample_index];
                                std::fprintf(
                                    timing_csv,
                                    "rocm,prefill,%s,%u,%u,%s,%d,%d,%d,%s,"
                                    "%zu,1,%.9f,%a,%d%s\n",
                                    format.name.c_str(),
                                    static_cast<unsigned>(codebook_id),
                                    static_cast<unsigned>(execution_codebook_id),
                                    shape.name.c_str(), M, shape.N, shape.K,
                                    variant.name.c_str(), sample_index,
                                    latency_us, latency_us,
                                    llaminar2::test::kDensePrefillCapturedOperations,
                                    native_csv_suffix.c_str());
                            }
                        }
                        rows.push_back(std::move(r));
                    }
                    ASSERT_FALSE(rows.empty()) << "No ROCm NativeQuantised variants selected.";

                    const auto best_it = std::min_element(
                        rows.begin(), rows.end(),
                        [](const BenchResult &lhs, const BenchResult &rhs)
                        {
                            const double lhs_time = lhs.correctness_pass && lhs.min_us > 0.0 ? lhs.min_us : 1e100;
                            const double rhs_time = rhs.correctness_pass && rhs.min_us > 0.0 ? rhs.min_us : 1e100;
                            return lhs_time < rhs_time;
                        });
                    ASSERT_NE(best_it, rows.end());

                    for (const auto &r : rows)
                    {
                        const int is_best = (&r == &(*best_it)) ? 1 : 0;
                        if (csv)
                        {
                            std::fprintf(csv,
                                         "rocm,prefill,%s,%u,%u,%s,%s,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.6f,%d,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d,%d,%s,%zu%s\n",
                                         format.name.c_str(),
                                         static_cast<unsigned>(codebook_id),
                                         static_cast<unsigned>(execution_codebook_id),
                                         shape.name.c_str(),
                                         shape.category.c_str(),
                                         M,
                                         shape.N,
                                         shape.K,
                                         r.variant_name.c_str(),
                                         r.min_us,
                                         r.mean_us,
                                         r.stddev_us,
                                         r.gflops,
                                         r.cosine_sim,
                                         r.correctness_pass ? 1 : 0,
                                         r.byte_mismatches_vs_auto,
                                         r.first_byte_mismatch_vs_auto,
                                         is_best,
                                         r.observed_n_tile,
                                         r.observed_m_tile,
                                         r.observed_min_blocks,
                                         r.observed_unroll,
                                         r.observed_full_tiles ? 1 : 0,
                                         r.registers_per_thread,
                                         r.local_memory_bytes_per_thread,
                                         r.static_shared_memory_bytes,
                                         r.max_threads_per_block,
                                         r.max_active_blocks_per_sm,
                                         vnniPrefillProducerName(r.observed_producer),
                                         r.dynamic_shared_memory_bytes,
                                         native_csv_suffix.c_str());
                            ++executed_rows;
                        }
                    }
                    if (csv)
                        std::fflush(csv);
                    if (timing_csv)
                        std::fflush(timing_csv);

                    std::fprintf(stderr,
                                 "[ROCmNativeQuantised][TRAINER][BEST] format=%s codebook=%u shape=%s M=%d variant=%s time_us=%.3f byte_mismatches=%zu\n",
                                 format.name.c_str(),
                                 static_cast<unsigned>(codebook_id),
                                 shape.name.c_str(),
                                 M,
                                 best_it->variant_name.c_str(),
                                 best_it->min_us,
                                 best_it->byte_mismatches_vs_auto);

                    ++executed_cases;
                }
            }
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(executed_rows, 0) << "ROCm NativeQuantised trainer CSV had no rows.";
        }
        if (timing_csv)
            ASSERT_EQ(std::fclose(timing_csv), 0);
        ASSERT_GT(executed_cases, 0) << "No ROCm NativeQuantised trainer cases selected.";
#endif
    }

} // namespace
