/**
 * @file Perf__NativeVNNI_Sweep.cpp
 * @brief Per-shape tuning sweep for native-VNNI GEMM prefill.
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
 *
 * @note Requires ROCm device. Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
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
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/ScopedGPUStream.h"
#include "../native_vnni_dispatch/GPUTrainerVerification.h"
#include "../native_vnni_dispatch/NativeVNNIShapeManifest.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
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
             llaminar2::test::native_vnni_dispatch::nativeVnniShapeManifest())
        {
            if (shape.role != "production" || !shape.exact_overlay)
                continue;
            result.push_back(
                {shape.name, shape.aspect_bucket, shape.N, shape.K});
        }
        if (result.empty())
            throw std::runtime_error(
                "ROCm NativeVNNI manifest has no production exact overlays");
        return result;
    }();

    // Decode and grouped verifier depths have dedicated common-trainer
    // transactions. This legacy harness is restricted to ordinary prefill.
    static const std::vector<int> M_VALUES = defaultPrefillGraphBucketSizes();

    struct FormatSpec
    {
        std::string name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    static const std::vector<FormatSpec> NVNNI_FORMATS = {
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

    static const NativeVnniFormatInfo &requireNativeVnniInfo(const TensorBase *weights, const std::string &format_name)
    {
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        const NativeVnniFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info)
            throw std::runtime_error("ROCm NativeVNNI sweep format " + format_name + " did not expose vnniFormatInfo()");
        return *info;
    }

    TEST(NativeVNNISweepOffline, ShapeAndFormatInventoriesAreCanonical)
    {
        const std::set<std::string> expected_formats = {
            "Q4_0", "IQ4_NL", "IQ4_XS", "Q4_1", "Q4_K", "Q5_0", "Q5_1",
            "Q5_K", "Q6_K", "Q3_K", "Q2_K", "IQ3_S", "IQ3_XXS", "IQ2_S",
            "IQ2_XS", "IQ2_XXS", "IQ1_S", "IQ1_M", "Q8_0", "Q8_1", "Q8_K"};
        std::set<std::string> observed_formats;
        for (const auto &format : NVNNI_FORMATS)
        {
            auto weights = format.create(/*N=*/2, /*K=*/256);
            ASSERT_NE(weights, nullptr) << format.name;
            (void)requireNativeVnniInfo(weights.get(), format.name);
            EXPECT_TRUE(observed_formats.insert(format.name).second)
                << "duplicate ROCm NativeVNNI source format " << format.name;
        }
        EXPECT_EQ(observed_formats, expected_formats);

        std::set<std::tuple<std::string, int, int>> expected_shapes;
        for (const auto &shape :
             llaminar2::test::native_vnni_dispatch::nativeVnniShapeManifest())
        {
            if (shape.role == "production" && shape.exact_overlay)
                expected_shapes.emplace(shape.name, shape.N, shape.K);
        }
        std::set<std::tuple<std::string, int, int>> observed_shapes;
        for (const auto &shape : GEMM_SHAPES)
            EXPECT_TRUE(
                observed_shapes.emplace(shape.name, shape.N, shape.K).second)
                << "duplicate ROCm NativeVNNI exact geometry " << shape.name;
        EXPECT_EQ(observed_shapes, expected_shapes);

        EXPECT_EQ(M_VALUES, llaminar2::defaultPrefillGraphBucketSizes());
        EXPECT_EQ(std::count(M_VALUES.begin(), M_VALUES.end(), 1), 0);
        for (int m = 2; m <= llaminar2::kDefaultNativeVNNIVerifierRowCapacity;
             ++m)
            EXPECT_EQ(std::count(M_VALUES.begin(), M_VALUES.end(), m), 0)
                << "M=" << m << " belongs to grouped verifier training";
    }

    // =========================================================================
    // Variant definitions
    // =========================================================================

    struct VariantConfig
    {
        std::string name;
        int force_n;    // -1=auto, 64=force N64, 128=force N128
        int mt;         // -1=auto, 16/32/64
        int min_blocks; // -1=auto, 1=bare, 2=2-wave, 3=3-wave
        int unroll;     // -1=auto, 0/1/2/4
        int full_tiles = 0; // -1=auto, 0=checked edges, 1=proven full tiles
    };

    static const std::vector<VariantConfig> NVNNI_VARIANTS = {
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
        {"Auto", -1, -1, -1, -1, -1},
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

    /** Return the exact candidate inventory launchable for one source format. */
    static std::vector<VariantConfig> trainerVariantsForFormat(
        const std::string &format_name)
    {
        std::vector<VariantConfig> result;
        result.reserve(NVNNI_VARIANTS.size() + 32);
        const auto auto_iterator = std::find_if(
            NVNNI_VARIANTS.begin(),
            NVNNI_VARIANTS.end(),
            [](const VariantConfig &variant)
            { return variant.name == "Auto"; });
        for (auto iterator = NVNNI_VARIANTS.begin();
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
        if (auto_iterator != NVNNI_VARIANTS.end())
            result.push_back(*auto_iterator);
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
        int registers_per_thread = 0;
        size_t local_memory_bytes_per_thread = 0;
        size_t static_shared_memory_bytes = 0;
        int max_threads_per_block = 0;
        int max_active_blocks_per_sm = 0;
    };

#ifdef HAVE_ROCM
    extern "C" bool rocmNativeVNNIPrefill_getLastLaunchSelection(
        uint8_t *codebook_id,
        int *n_tile,
        int *m_tile,
        int *min_blocks,
        int *unroll,
        bool *full_tiles);
    extern "C" bool rocmNativeVNNIPrefill_getLastLaunchResources(
        int *registers_per_thread,
        size_t *local_memory_bytes_per_thread,
        size_t *static_shared_memory_bytes,
        int *max_threads_per_block,
        int *max_active_blocks_per_sm);
#endif

#ifdef HAVE_ROCM
    /**
     * @brief Typed, process-local NativeVNNI candidate override for the trainer.
     *
     * The production launcher reads its tuning controls from the typed
     * `DebugEnv` snapshot. Rewriting environment variables and reparsing the
     * complete ROCm configuration for every launch made the old trainer both
     * noisy and needlessly expensive. This scope changes only the five launch
     * fields owned by the NativeVNNI tournament and restores them on teardown.
     * The turnkey collector gives every physical GPU its own process, so these
     * process-local controls cannot race another candidate lane.
     */
    class ScopedROCmNativeVNNITrainerOverride
    {
    public:
        ScopedROCmNativeVNNITrainerOverride()
        {
            const auto &config = debugEnv().rocm;
            saved_mt_ = config.nvnni_mt;
            saved_unroll_ = config.nvnni_unroll;
            saved_min_blocks_ = config.nvnni_min_blocks;
            saved_force_n64_ = config.nvnni_force_n64;
            saved_force_n128_ = config.nvnni_force_n128;
            saved_full_tiles_ = config.nvnni_full_tiles;
        }

        ~ScopedROCmNativeVNNITrainerOverride()
        {
            auto &config = mutableDebugEnv().rocm;
            config.nvnni_mt = saved_mt_;
            config.nvnni_unroll = saved_unroll_;
            config.nvnni_min_blocks = saved_min_blocks_;
            config.nvnni_force_n64 = saved_force_n64_;
            config.nvnni_force_n128 = saved_force_n128_;
            config.nvnni_full_tiles = saved_full_tiles_;
        }

        ScopedROCmNativeVNNITrainerOverride(
            const ScopedROCmNativeVNNITrainerOverride &) = delete;
        ScopedROCmNativeVNNITrainerOverride &operator=(
            const ScopedROCmNativeVNNITrainerOverride &) = delete;

        /** Select one exact compile-time candidate for subsequent launches. */
        void apply(const VariantConfig &variant)
        {
            auto &config = mutableDebugEnv().rocm;
            config.nvnni_mt = variant.mt;
            config.nvnni_unroll = variant.unroll;
            config.nvnni_min_blocks = variant.min_blocks;
            config.nvnni_force_n64 = variant.force_n == 64;
            config.nvnni_force_n128 = variant.force_n == 128;
            config.nvnni_full_tiles = variant.full_tiles;
        }

    private:
        int saved_mt_ = -1;
        int saved_unroll_ = -1;
        int saved_min_blocks_ = -1;
        bool saved_force_n64_ = false;
        bool saved_force_n128_ = false;
        int saved_full_tiles_ = -1;
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
        PreparedROCmSweepExecution(
            const GEMMShape &shape,
            int m,
            TensorBase *weights,
            const std::vector<VariantConfig> &variants,
            int maximum_bench_runs)
            : shape_(shape), m_(m)
        {
            if (!weights || m_ <= 0 || shape_.N <= 0 || shape_.K <= 0 ||
                maximum_bench_runs <= 0)
            {
                throw std::invalid_argument(
                    "invalid ROCm NativeVNNI tournament geometry");
            }
            requireHip(hipSetDevice(kDeviceOrdinal), "select device");
            expected_codebook_ =
                requireNativeVnniInfo(weights, "ROCm tournament source weights")
                    .codebook_id;
            if (!packWeightsToROCm(weights, packed_))
                throw std::runtime_error(
                    "failed to pack persistent ROCm NativeVNNI weights");

            kernel_ = std::make_unique<ROCmQuantisedGemmKernel>(
                &packed_, kDeviceOrdinal);
            stream_owner_ = std::make_unique<ScopedGPUStream>(
                DeviceId::rocm(kDeviceOrdinal));
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
                DeviceId::rocm(kDeviceOrdinal), budget);
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

            const DeviceId device = DeviceId::rocm(kDeviceOrdinal);
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
            int bench_runs)
        {
            if (warmup_runs <= 0 || bench_runs <= 0 ||
                static_cast<size_t>(bench_runs) > start_events_.size())
            {
                throw std::invalid_argument(
                    "invalid ROCm NativeVNNI timing repetition count");
            }

            override_.apply(variant);
            BenchResult result;
            result.variant_name = variant.name;
            result.shape_name = shape_.name;
            result.category = shape_.category;
            result.M = m_;
            result.N = shape_.N;
            result.K = shape_.K;

            // Resolve immutable compiler resources before the timing batch.
            // This single untimed launch also proves the requested template is
            // launchable for the exact cell geometry. A spilling candidate is
            // rejected here and never contributes a benchmark observation.
            launch(output_.get());
            checkStream("candidate resource probe");
            uint8_t observed_codebook = 0;
            if (!rocmNativeVNNIPrefill_getLastLaunchSelection(
                    &observed_codebook,
                    &result.observed_n_tile,
                    &result.observed_m_tile,
                    &result.observed_min_blocks,
                    &result.observed_unroll,
                    &result.observed_full_tiles) ||
                observed_codebook != expected_codebook_ ||
                !rocmNativeVNNIPrefill_getLastLaunchResources(
                    &result.registers_per_thread,
                    &result.local_memory_bytes_per_thread,
                    &result.static_shared_memory_bytes,
                    &result.max_threads_per_block,
                    &result.max_active_blocks_per_sm))
            {
                throw std::runtime_error(
                    "ROCm NativeVNNI trainer could not inspect its launch");
            }
            if (result.local_memory_bytes_per_thread != 0)
            {
                throw std::runtime_error(
                    "ROCm NativeVNNI candidate spills local memory: " +
                    variant.name + " bytes_per_thread=" +
                    std::to_string(result.local_memory_bytes_per_thread));
            }

            for (int iteration = 0; iteration < warmup_runs; ++iteration)
                launch(output_.get());
            checkStream("candidate warmup");

            for (int iteration = 0; iteration < bench_runs; ++iteration)
            {
                const size_t index = static_cast<size_t>(iteration);
                requireHip(
                    hipEventRecord(start_events_[index].get(), stream_),
                    "record candidate start event");
                launch(output_.get());
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
                    static_cast<double>(elapsed_ms) * 1000.0;
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

    private:
        static constexpr int kDeviceOrdinal = 0;

        /** Return the production exact-M route used as the byte oracle. */
        static const VariantConfig &autoVariant()
        {
            const auto iterator = std::find_if(
                NVNNI_VARIANTS.begin(),
                NVNNI_VARIANTS.end(),
                [](const VariantConfig &variant)
                { return variant.name == "Auto"; });
            if (iterator == NVNNI_VARIANTS.end())
                throw std::logic_error(
                    "ROCm NativeVNNI candidate inventory has no Auto route");
            return *iterator;
        }

        static void requireHip(hipError_t error, const char *operation)
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("ROCm NativeVNNI sweep failed to ") +
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
                    "ROCm NativeVNNI production launch failed");
            }
        }

        void checkStream(const char *operation)
        {
            requireHip(hipStreamSynchronize(stream_), operation);
        }

        void buildExactMPrefillOracle()
        {
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
                    "ROCm NativeVNNI full-buffer byte comparison failed");
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

        const GEMMShape &shape_;
        int m_ = 0;
        uint8_t expected_codebook_ = 0;
        ScopedROCmNativeVNNITrainerOverride override_;
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

    class NativeVNNISweepTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifdef HAVE_ROCM
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            has_device_ = (err == hipSuccess && device_count > 0);
            if (has_device_)
            {
                (void)hipSetDevice(0);
                hipDeviceProp_t props;
                if (hipGetDeviceProperties(&props, 0) == hipSuccess)
                    device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
                else
                    device_name_ = "rocm:0";
            }
#else
            has_device_ = false;
#endif
        }

        bool has_device_ = false;
        std::string device_name_;
    };

    // =========================================================================
    // Test: Full sweep — all variants × all shapes × all M values
    // =========================================================================

    TEST_F(NativeVNNISweepTest, Q4_0_VariantSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[Native-VNNI Q4_0 Sweep] Variant Sweep\n");
        fprintf(stderr, "[Native-VNNI Q4_0 Sweep] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[Native-VNNI Q4_0 Sweep] %zu shapes × %zu M values × %zu variants\n",
                GEMM_SHAPES.size(), M_VALUES.size(), NVNNI_VARIANTS.size());
        fprintf(stderr, "[Native-VNNI Q4_0 Sweep] %d warmup + %d timed runs each\n",
                WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr,
                "[Native-VNNI Q4_0 Sweep] Correctness gate: full output byte equality vs Auto\n\n");

        const size_t num_shapes = GEMM_SHAPES.size();
        const size_t num_m = M_VALUES.size();
        const size_t num_variants = NVNNI_VARIANTS.size();

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
                    NVNNI_VARIANTS,
                    BENCH_RUNS);

                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &variant = NVNNI_VARIANTS[vi];
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
            for (const auto &v : NVNNI_VARIANTS)
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

                    if (NVNNI_VARIANTS[vi].name == "Auto")
                        auto_us = res.min_us;
                }

                // Best variant name
                if (best_idx >= 0)
                    table << NVNNI_VARIANTS[static_cast<size_t>(best_idx)].name;
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
                    if (NVNNI_VARIANTS[vi].name == "Auto")
                        auto_us = res.min_us;
                }

                const char *best_name = (best_idx >= 0) ? NVNNI_VARIANTS[static_cast<size_t>(best_idx)].name.c_str() : "N/A";
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

    TEST_F(NativeVNNISweepTest, TrainerCsv_CodebookTagged)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        std::set<std::string> format_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_SWEEP_FORMATS");
        if (format_filters.empty())
            format_filters.insert("q4_0");
        const std::set<std::string> shape_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_SWEEP_SHAPES");
        const std::set<std::string> variant_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_SWEEP_VARIANTS");
        const std::vector<int> m_values = getEnvCsvInts(
            "LLAMINAR_ROCM_NVNNI_SWEEP_M",
            defaultPrefillGraphBucketSizes());
        const int warmup_runs = std::max(
            1,
            getEnvInt("LLAMINAR_ROCM_NVNNI_SWEEP_WARMUP")
                .value_or(WARMUP_RUNS));
        const int bench_runs = std::max(
            1,
            getEnvInt("LLAMINAR_ROCM_NVNNI_SWEEP_BENCH")
                .value_or(BENCH_RUNS));
        const int max_cases = std::max(1, getEnvInt("LLAMINAR_ROCM_NVNNI_SWEEP_MAX_CASES").value_or(1));
        const std::string csv_path = getEnvString("LLAMINAR_ROCM_NVNNI_SWEEP_CSV");
        const std::string timing_csv_path =
            getEnvString("LLAMINAR_ROCM_NVNNI_SWEEP_TIMING_CSV");

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open ROCm NativeVNNI sweep CSV: " << csv_path;
            std::fprintf(csv,
                         "backend,phase,format,codebook,shape,category,m,n,k,variant,min_us,mean_us,stddev_us,gflops,cosine,correctness_pass,byte_mismatches_vs_auto,first_byte_mismatch_vs_auto,is_best,observed_n_tile,observed_m_tile,observed_min_blocks,observed_unroll,observed_full_tiles,registers_per_thread,local_memory_bytes_per_thread,static_shared_memory_bytes,max_threads_per_block,max_active_blocks_per_sm\n");
        }

        std::FILE *timing_csv = nullptr;
        if (!timing_csv_path.empty())
        {
            timing_csv = std::fopen(timing_csv_path.c_str(), "w");
            ASSERT_NE(timing_csv, nullptr)
                << "Failed to open ROCm NativeVNNI timing CSV: "
                << timing_csv_path;
            std::fprintf(
                timing_csv,
                "backend,phase,format,codebook,shape,m,n,k,variant,"
                "sample_index,timed_replays,latency_us,latency_us_hex\n");
        }

        int executed_cases = 0;
        int executed_rows = 0;

        for (const auto &format : NVNNI_FORMATS)
        {
            if (!shouldRunName(format_filters, format.name))
                continue;

            const std::vector<VariantConfig> launchable_variants =
                trainerVariantsForFormat(format.name);
            std::vector<VariantConfig> selected_variants;
            std::copy_if(
                launchable_variants.begin(),
                launchable_variants.end(),
                std::back_inserter(selected_variants),
                [&variant_filters](const VariantConfig &variant)
                { return shouldRunName(variant_filters, variant.name); });
            ASSERT_FALSE(selected_variants.empty())
                << "No ROCm NativeVNNI variants selected for " << format.name;

            for (const auto &shape : GEMM_SHAPES)
            {
                if (!shouldRunName(shape_filters, shape.name))
                    continue;
                if ((shape.K % 32) != 0)
                    continue;

                auto weights = format.create(static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
                const uint8_t codebook_id = requireNativeVnniInfo(weights.get(), format.name).codebook_id;

                for (const int M : m_values)
                {
                    if (executed_cases >= max_cases)
                        break;

                    PreparedROCmSweepExecution execution(
                        shape,
                        M,
                        weights.get(),
                        selected_variants,
                        bench_runs);

                    std::vector<BenchResult> rows;
                    rows.reserve(selected_variants.size());
                    for (const auto &variant : selected_variants)
                    {
                        auto r = execution.measure(
                            variant,
                            warmup_runs,
                            bench_runs);
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
                                    "rocm,prefill,%s,%u,%s,%d,%d,%d,%s,"
                                    "%zu,1,%.9f,%a\n",
                                    format.name.c_str(),
                                    static_cast<unsigned>(codebook_id),
                                    shape.name.c_str(), M, shape.N, shape.K,
                                    variant.name.c_str(), sample_index,
                                    latency_us, latency_us);
                            }
                        }
                        rows.push_back(std::move(r));
                    }
                    ASSERT_FALSE(rows.empty()) << "No ROCm NativeVNNI variants selected.";

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
                                         "rocm,prefill,%s,%u,%s,%s,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.6f,%d,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d,%d\n",
                                         format.name.c_str(),
                                         static_cast<unsigned>(codebook_id),
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
                                         r.max_active_blocks_per_sm);
                            ++executed_rows;
                        }
                    }
                    if (csv)
                        std::fflush(csv);
                    if (timing_csv)
                        std::fflush(timing_csv);

                    std::fprintf(stderr,
                                 "[ROCmNativeVNNI][TRAINER][BEST] format=%s codebook=%u shape=%s M=%d variant=%s time_us=%.3f byte_mismatches=%zu\n",
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
            ASSERT_GT(executed_rows, 0) << "ROCm NativeVNNI trainer CSV had no rows.";
        }
        if (timing_csv)
            ASSERT_EQ(std::fclose(timing_csv), 0);
        ASSERT_GT(executed_cases, 0) << "No ROCm NativeVNNI trainer cases selected.";
#endif
    }

} // namespace
