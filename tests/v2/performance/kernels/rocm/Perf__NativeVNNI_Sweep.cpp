/**
 * @file Perf__NativeVNNI_Sweep.cpp
 * @brief Per-shape tuning sweep for native-VNNI GEMM prefill.
 *
 * Benchmarks all combinations of:
 *   - N_TILE: 64 vs 128
 *   - M_TILE: {16, 32, 64}
 *   - MIN_BLOCKS: {1 (bare launch_bounds), 2 (default 2-wave)}
 *   - UNROLL_G: {0 (no hint), 2, 4 (default)}
 *   - Auto dispatch (current heuristic baseline)
 *
 * Shapes tested: Qwen2.5/Qwen3.6 dense and MoE-style production shapes.
 * M values: MTP small rows plus the canonical graph-prefill bucket policy.
 *
 * Output: per-shape best variant table with correctness gate (cosine > 0.9990).
 *
 * @note Requires ROCm device. Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PrefillGraphBucketDefaults.h"
#include "../../../utils/TestTensorFactory.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "GpuVerification.h"
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
#ifdef HAVE_ROCM
    using gpu_verify::destroyAllHipBLAS;
    using gpu_verify::gpuCosineSimilarity;
    using gpu_verify::gpuReferenceFP32Gemm;
    using gpu_verify::GpuWeightsCache;
#endif

    // =========================================================================
    // Constants
    // =========================================================================

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;
    constexpr float COSINE_SIM_GATE = 0.9990f;

    // =========================================================================
    // Scoped environment override
    // =========================================================================

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const std::string &value)
            : name_(name)
        {
            const char *existing = std::getenv(name_);
            had_original_ = (existing != nullptr);
            if (had_original_)
                original_value_ = existing;
            ::setenv(name_, value.c_str(), 1);
            mutableDebugEnv().rocm.reload();
        }

        ~ScopedEnvOverride()
        {
            if (had_original_)
                ::setenv(name_, original_value_.c_str(), 1);
            else
                ::unsetenv(name_);
            mutableDebugEnv().rocm.reload();
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        const char *name_;
        bool had_original_ = false;
        std::string original_value_;
    };

    // =========================================================================
    // Shape definitions (Qwen2.5 0.5B / 3B / 7B)
    // =========================================================================

    struct GEMMShape
    {
        std::string name;
        std::string category;
        int N;
        int K;
    };

    static const std::vector<GEMMShape> GEMM_SHAPES = {
        // Qwen3.5/Qwen3.6 MoE expert FFN shapes.
        {"35BMoE_Expert_GateUp", "MoE", 512, 2048},
        {"35BMoE_Expert_Down", "MoE", 2048, 512},
        // Qwen2.5-0.5B (hidden=896, intermediate=4864)
        {"0.5B_AttnQKV", "Attention", 2688, 896},
        {"0.5B_AttnOut", "Attention", 896, 896},
        {"0.5B_FFN_Up", "FFN_Up", 4864, 896},
        {"0.5B_FFN_Dn", "FFN_Down", 896, 4864},
        {"0.5B_LM_Head", "LM_Head", 151936, 896},
        // Qwen2.5-3B (hidden=2048, intermediate=11008)
        {"3B_AttnQKV", "Attention", 6144, 2048},
        {"3B_AttnOut", "Attention", 2048, 2048},
        {"3B_FFN_Up", "FFN_Up", 11008, 2048},
        {"3B_FFN_Dn", "FFN_Down", 2048, 11008},
        {"3B_LM_Head", "LM_Head", 151936, 2048},
        // Qwen2.5-7B (hidden=3584, intermediate=18944)
        {"7B_AttnQKV", "Attention", 10752, 3584},
        {"7B_AttnOut", "Attention", 3584, 3584},
        {"7B_FFN_Up", "FFN_Up", 18944, 3584},
        {"7B_FFN_Dn", "FFN_Down", 3584, 18944},
        {"7B_LM_Head", "LM_Head", 151936, 3584},
        // Qwen3.6 27B dense / hybrid GDN production shapes.
        {"Qwen36_Attn_Q", "Attention", 5120, 5120},
        {"Qwen36_Attn_KV", "Attention", 1024, 5120},
        {"Qwen36_Attn_Wo", "Attention", 5120, 5120},
        {"Qwen36_FFN_GateUp", "FFN_Up", 17408, 5120},
        {"Qwen36_FFN_DownProjection", "FFN_Down", 5120, 17408},
        {"Qwen36_GDN_InnerProjection", "GDN", 10240, 5120},
        {"Qwen36_GDN_ZProjection", "GDN", 6144, 5120},
        {"Qwen36_GDN_TimeProjection", "GDN", 1024, 5120},
        {"Qwen36_GDN_OutputProjection", "GDN", 5120, 6144},
        {"Qwen36_LM_Head", "LM_Head", 248320, 5120},
    };

    static const std::vector<int> M_VALUES = defaultNativeVNNIDispatchTrainingRows();

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
        {"N128/MT64/MB1", 128, 64, 1, -1},
        {"N128/MT64/MB2", 128, 64, 2, -1},
        // UNROLL_G sweep on best known config (N64/MT64/MB1 — most likely winner)
        {"N64/MT64/MB1/U0", 64, 64, 1, 0},
        {"N64/MT64/MB1/U1", 64, 64, 1, 1},
        {"N64/MT64/MB1/U2", 64, 64, 1, 2},
        {"N64/MT64/MB1/U4", 64, 64, 1, 4},
        // UNROLL_G sweep on N128/MT32/MB1
        {"N128/MT32/MB1/U0", 128, 32, 1, 0},
        {"N128/MT32/MB1/U1", 128, 32, 1, 1},
        {"N128/MT32/MB1/U2", 128, 32, 1, 2},
        {"N128/MT32/MB1/U4", 128, 32, 1, 4},
        // 3-wave variants for comparison
        {"N64/MT16/MB3", 64, 16, 3, -1},
        {"N128/MT16/MB3", 128, 16, 3, -1},
        // Auto dispatch (current heuristic)
        {"Auto", -1, -1, -1, -1},
    };

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
    };

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
                hipGetDeviceProperties(&props, 0);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
            }
#else
            has_device_ = false;
#endif
        }

        void TearDown() override
        {
#ifdef HAVE_ROCM
            destroyAllHipBLAS();
#endif
        }

        bool has_device_ = false;
        std::string device_name_;

#ifdef HAVE_ROCM

        BenchResult benchmarkVariant(const VariantConfig &variant,
                                     const GEMMShape &shape, int M,
                                     TensorBase *weights,
                                     const GpuWeightsCache &gpu_weights)
        {
            BenchResult result;
            result.variant_name = variant.name;
            result.shape_name = shape.name;
            result.category = shape.category;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            if (!weights)
                return result;

            // Set environment overrides for this variant
            ScopedEnvOverride force_n64("LLAMINAR_ROCM_NVNNI_FORCE_N64",
                                        variant.force_n == 64 ? "1" : "0");
            ScopedEnvOverride force_n128("LLAMINAR_ROCM_NVNNI_FORCE_N128",
                                         variant.force_n == 128 ? "1" : "0");
            ScopedEnvOverride mt_env("LLAMINAR_ROCM_NVNNI_MT",
                                     std::to_string(variant.mt));
            ScopedEnvOverride mb_env("LLAMINAR_ROCM_NVNNI_MIN_BLOCKS",
                                     std::to_string(variant.min_blocks));
            ScopedEnvOverride ug_env("LLAMINAR_ROCM_NVNNI_UNROLL",
                                     std::to_string(variant.unroll));

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights, packed))
                return result;

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            if (!workspace->allocate(reqs))
                return result;
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(0)))
            {
                kernel.unbindWorkspace();
                return result;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(0)))
            {
                kernel.unbindWorkspace();
                return result;
            }

            // Correctness: compare against hipBLAS FP32 reference
            {
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                (void)hipDeviceSynchronize();
                output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

                if (gpu_weights.d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input.get());
                    const float *d_input = reinterpret_cast<const float *>(in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        const size_t out_elems = static_cast<size_t>(M) * shape.N;
                        float *d_ref = nullptr;
                        if (hipMalloc(&d_ref, out_elems * sizeof(float)) == hipSuccess)
                        {
                            if (gpuReferenceFP32Gemm(d_input, gpu_weights.d_weights,
                                                     d_ref, M, shape.N, shape.K, 0))
                            {
                                (void)hipDeviceSynchronize();
                                const float *d_out = reinterpret_cast<const float *>(
                                    dynamic_cast<FP32Tensor *>(output.get())->gpu_data_ptr());
                                result.cosine_sim = gpuCosineSimilarity(d_out, d_ref, out_elems, 0);
                                result.correctness_pass = (result.cosine_sim >= COSINE_SIM_GATE);
                            }
                            (void)hipFree(d_ref);
                        }
                    }
                }
            }

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
            (void)hipDeviceSynchronize();

            // Timed runs using HIP events
            hipEvent_t start = nullptr, stop = nullptr;
            (void)hipEventCreate(&start);
            (void)hipEventCreate(&stop);

            std::vector<double> times_us;
            times_us.reserve(BENCH_RUNS);

            for (int i = 0; i < BENCH_RUNS; ++i)
            {
                (void)hipDeviceSynchronize();
                (void)hipEventRecord(start);
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                (void)hipEventRecord(stop);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);

            std::sort(times_us.begin(), times_us.end());

            result.min_us = times_us.front();
            result.mean_us = std::accumulate(times_us.begin(), times_us.end(), 0.0) /
                             static_cast<double>(times_us.size());
            double sq_sum = 0.0;
            for (double t : times_us)
                sq_sum += (t - result.mean_us) * (t - result.mean_us);
            result.stddev_us = std::sqrt(sq_sum / static_cast<double>(times_us.size()));

            double flops = 2.0 * M * static_cast<double>(shape.N) * shape.K;
            result.gflops = flops / (result.min_us * 1e-6) / 1e9;

            kernel.unbindWorkspace();
            return result;
        }

#endif
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
        fprintf(stderr, "[Native-VNNI Q4_0 Sweep] Correctness gate: cosine >= %.4f\n\n",
                COSINE_SIM_GATE);

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

            GpuWeightsCache gpu_weights;
            if (q4_weights)
            {
                const float *w_fp32 = q4_weights->data();
                if (w_fp32)
                    gpu_weights.upload(w_fp32, shape.N, shape.K, 0);
            }

            for (size_t mi = 0; mi < num_m; ++mi)
            {
                int M = M_VALUES[mi];

                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &variant = NVNNI_VARIANTS[vi];
                    auto r = benchmarkVariant(variant, shape, M,
                                              q4_weights.get(), gpu_weights);
                    results[si * num_m * num_variants + mi * num_variants + vi] = std::move(r);
                    ++completed;

                    const auto &res = results[si * num_m * num_variants + mi * num_variants + vi];
                    fprintf(stderr, "  %s %s M=%d: %.0f μs (cos=%.4f) [%d/%d]\n",
                            variant.name.c_str(), shape.name.c_str(), M,
                            res.min_us, res.cosine_sim,
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
            defaultNativeVNNIDispatchTrainingRows());
        const int max_cases = std::max(1, getEnvInt("LLAMINAR_ROCM_NVNNI_SWEEP_MAX_CASES").value_or(1));
        const std::string csv_path = getEnvString("LLAMINAR_ROCM_NVNNI_SWEEP_CSV");

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open ROCm NativeVNNI sweep CSV: " << csv_path;
            std::fprintf(csv,
                         "backend,phase,format,codebook,shape,category,m,n,k,variant,min_us,mean_us,stddev_us,gflops,cosine,correctness_pass,is_best\n");
        }

        int executed_cases = 0;
        int executed_rows = 0;

        for (const auto &format : NVNNI_FORMATS)
        {
            if (!shouldRunName(format_filters, format.name))
                continue;

            for (const auto &shape : GEMM_SHAPES)
            {
                if (!shouldRunName(shape_filters, shape.name))
                    continue;
                if ((shape.K % 32) != 0)
                    continue;

                auto weights = format.create(static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
                const uint8_t codebook_id = requireNativeVnniInfo(weights.get(), format.name).codebook_id;

                GpuWeightsCache gpu_weights;
                if (weights)
                {
                    const float *w_fp32 = weights->data();
                    if (w_fp32)
                        gpu_weights.upload(w_fp32, shape.N, shape.K, 0);
                }

                for (const int M : m_values)
                {
                    if (executed_cases >= max_cases)
                        break;

                    std::vector<BenchResult> rows;
                    rows.reserve(NVNNI_VARIANTS.size());
                    for (const auto &variant : NVNNI_VARIANTS)
                    {
                        if (!shouldRunName(variant_filters, variant.name))
                            continue;
                        auto r = benchmarkVariant(variant, shape, M, weights.get(), gpu_weights);
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
                                         "rocm,prefill,%s,%u,%s,%s,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.6f,%d,%d\n",
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
                                         is_best);
                            ++executed_rows;
                        }
                    }
                    if (csv)
                        std::fflush(csv);

                    std::fprintf(stderr,
                                 "[ROCmNativeVNNI][TRAINER][BEST] format=%s codebook=%u shape=%s M=%d variant=%s time_us=%.3f cosine=%.6f\n",
                                 format.name.c_str(),
                                 static_cast<unsigned>(codebook_id),
                                 shape.name.c_str(),
                                 M,
                                 best_it->variant_name.c_str(),
                                 best_it->min_us,
                                 best_it->cosine_sim);

                    ++executed_cases;
                }
            }
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(executed_rows, 0) << "ROCm NativeVNNI trainer CSV had no rows.";
        }
        ASSERT_GT(executed_cases, 0) << "No ROCm NativeVNNI trainer cases selected.";
#endif
    }

} // namespace
