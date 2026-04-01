/**
 * @file Perf__NativeVNNI_Sweep.cpp
 * @brief Per-shape tuning sweep for native-VNNI GEMM (Q4_0 prefill).
 *
 * Benchmarks all combinations of:
 *   - N_TILE: 64 vs 128
 *   - M_TILE: {16, 32, 64}
 *   - MIN_BLOCKS: {1 (bare launch_bounds), 2 (default 2-wave)}
 *   - UNROLL_G: {0 (no hint), 2, 4 (default)}
 *   - Auto dispatch (current heuristic baseline)
 *
 * Shapes tested: Qwen2.5-0.5B, 3B, 7B (Attention, FFN_Up, FFN_Down, LM_Head)
 * M values: 128, 256, 512 (typical prefill sizes)
 *
 * Output: per-shape best variant table with correctness gate (cosine > 0.9990).
 *
 * @note Requires ROCm device. Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
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
    };

    static const std::vector<int> M_VALUES = {128, 256, 512};

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

} // namespace
