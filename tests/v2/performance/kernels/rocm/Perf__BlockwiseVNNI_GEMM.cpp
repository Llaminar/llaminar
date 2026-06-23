/**
 * @file Perf__BlockwiseVNNI_GEMM.cpp
 * @brief Per-shape tuning sweep for blockwise activation-quantized INT8 VNNI GEMM.
 *
 * Benchmarks all combinations of:
 *   - Kernel variant: V3 (N_TILE=64) vs V7 (N_TILE=128)
 *   - M_TILE: {16, 32, 64} per variant
 *   - Row-wise (non-blockwise) baseline for comparison
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
    // Scoped environment override (same pattern as PrefillDispatchComparison)
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
    // Shape definitions
    // =========================================================================

    struct GEMMShape
    {
        std::string name;
        std::string category; // "Attention", "FFN_Up", "FFN_Down", "LM_Head"
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
        bool force_v3; // Force V3 for all shapes (LLAMINAR_ROCM_BLOCKWISE_FORCE_V3)
        bool force_v7; // Force V7 for all shapes (LLAMINAR_ROCM_BLOCKWISE_FORCE_V7)
        int v3_mt;     // V3 M_TILE override (-1=auto)
        int v7_mt;     // V7 M_TILE override (-1=auto)
        int v3_unroll; // V3 UNROLL_KK override (-1=auto, 0=disable, 1=full, 2/4=partial)
        int v7_unroll; // V7 UNROLL_KK override (-1=auto, 0=disable, 1=full, 2/4=partial)
    };

    // All blockwise variant configurations to sweep
    static const std::vector<VariantConfig> BLOCKWISE_VARIANTS = {
        // V3 variants (N_TILE=64, LDS double-buffered) — M_TILE sweep
        {"V3/MT16", true, false, 16, -1, -1, -1},
        {"V3/MT32", true, false, 32, -1, -1, -1},
        {"V3/MT64", true, false, 64, -1, -1, -1},
        // V3 unroll sweep (MT32 — default for most shapes)
        {"V3/MT32/U0", true, false, 32, -1, 0, -1},
        {"V3/MT32/U1", true, false, 32, -1, 1, -1},
        {"V3/MT32/U2", true, false, 32, -1, 2, -1},
        {"V3/MT32/U4", true, false, 32, -1, 4, -1},
        // V3 unroll sweep (MT64 — large M)
        {"V3/MT64/U0", true, false, 64, -1, 0, -1},
        {"V3/MT64/U1", true, false, 64, -1, 1, -1},
        {"V3/MT64/U2", true, false, 64, -1, 2, -1},
        {"V3/MT64/U4", true, false, 64, -1, 4, -1},
        // V7 variants (N_TILE=128, safe-tile split) — M_TILE sweep
        {"V7/MT16", false, true, -1, 16, -1, -1},
        {"V7/MT32", false, true, -1, 32, -1, -1},
        {"V7/MT64", false, true, -1, 64, -1, -1},
        // V7 unroll sweep (MT64 — default)
        {"V7/MT64/U0", false, true, -1, 64, -1, 0},
        {"V7/MT64/U1", false, true, -1, 64, -1, 1},
        {"V7/MT64/U2", false, true, -1, 64, -1, 2},
        {"V7/MT64/U4", false, true, -1, 64, -1, 4},
        // Auto dispatch (default: K>=N → V3/MT32, K<N → V7/MT64)
        {"Auto", false, false, -1, -1, -1, -1},
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

    class BlockwiseVNNIGEMMPerfTest : public ::testing::Test
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
                (void)hipGetDeviceProperties(&props, 0);
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

        /// Benchmark a single shape + M with a specific variant configuration.
        /// Uses multiply_tensor() which includes activation quantization + GEMM + epilogue.
        BenchResult benchmarkVariant(const VariantConfig &variant,
                                     const GEMMShape &shape, int M,
                                     const GpuWeightsCache &gpu_weights)
        {
            BenchResult result;
            result.variant_name = variant.name;
            result.shape_name = shape.name;
            result.category = shape.category;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            // Set environment overrides for this variant
            ScopedEnvOverride force_v3("LLAMINAR_ROCM_BLOCKWISE_FORCE_V3",
                                       variant.force_v3 ? "1" : "0");
            ScopedEnvOverride force_v7("LLAMINAR_ROCM_BLOCKWISE_FORCE_V7",
                                       variant.force_v7 ? "1" : "0");
            ScopedEnvOverride v3_mt("LLAMINAR_ROCM_BLOCKWISE_V3_MT",
                                    std::to_string(variant.v3_mt));
            ScopedEnvOverride v7_mt("LLAMINAR_ROCM_BLOCKWISE_V7_MT",
                                    std::to_string(variant.v7_mt));
            ScopedEnvOverride v3_unroll("LLAMINAR_ROCM_BLOCKWISE_V3_UNROLL",
                                        std::to_string(variant.v3_unroll));
            ScopedEnvOverride v7_unroll("LLAMINAR_ROCM_BLOCKWISE_V7_UNROLL",
                                        std::to_string(variant.v7_unroll));

            // Create Q8_0 weights and pack
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return result;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
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

            // Timed runs using HIP events for accurate GPU timing
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

            // Stats
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

        /// Benchmark non-blockwise (row-wise) baseline for comparison.
        /// Uses the same multiply_tensor() path but forces row-wise activation quant.
        BenchResult benchmarkRowWiseBaseline(const GEMMShape &shape, int M,
                                             const GpuWeightsCache &gpu_weights)
        {
            BenchResult result;
            result.variant_name = "Blockwise";
            result.shape_name = shape.name;
            result.category = shape.category;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return result;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
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

            // Correctness
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

            // Timed runs
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

    TEST_F(BlockwiseVNNIGEMMPerfTest, BlockwiseVariantSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[Blockwise VNNI GEMM] Variant Sweep\n");
        fprintf(stderr, "[Blockwise VNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[Blockwise VNNI GEMM] %zu shapes × %zu M values × %zu variants + row-wise baseline\n",
                GEMM_SHAPES.size(), M_VALUES.size(), BLOCKWISE_VARIANTS.size());
        fprintf(stderr, "[Blockwise VNNI GEMM] %d warmup + %d timed runs each\n",
                WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr, "[Blockwise VNNI GEMM] Correctness gate: cosine >= %.4f\n\n",
                COSINE_SIM_GATE);

        // Results: [shape_idx][m_idx][variant_idx] + [shape_idx][m_idx] for row-wise
        const size_t num_shapes = GEMM_SHAPES.size();
        const size_t num_m = M_VALUES.size();
        const size_t num_variants = BLOCKWISE_VARIANTS.size();

        // 3D array: shape × M × variant
        std::vector<BenchResult> bw_results(num_shapes * num_m * num_variants);
        // 2D array: shape × M for row-wise baseline
        std::vector<BenchResult> rw_results(num_shapes * num_m);

        int total_benchmarks = static_cast<int>(num_shapes * num_m * (num_variants + 1));
        int completed = 0;

        for (size_t si = 0; si < num_shapes; ++si)
        {
            const auto &shape = GEMM_SHAPES[si];

            // Create GPU FP32 weights cache for this shape (shared across variants)
            GpuWeightsCache gpu_weights;
            {
                auto fp32_weights = TestTensorFactory::createQ8_0Random(
                    {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
                if (fp32_weights)
                {
                    const float *w_fp32 = fp32_weights->data();
                    if (w_fp32)
                        gpu_weights.upload(w_fp32, shape.N, shape.K, 0);
                }
            }

            for (size_t mi = 0; mi < num_m; ++mi)
            {
                int M = M_VALUES[mi];

                // Row-wise baseline
                {
                    auto r = benchmarkRowWiseBaseline(shape, M, gpu_weights);
                    rw_results[si * num_m + mi] = std::move(r);
                    ++completed;
                    fprintf(stderr, "  RowWise %s M=%d: %.0f μs (cos=%.4f) [%d/%d]\n",
                            shape.name.c_str(), M,
                            rw_results[si * num_m + mi].min_us,
                            rw_results[si * num_m + mi].cosine_sim,
                            completed, total_benchmarks);
                }

                // All blockwise variants
                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &variant = BLOCKWISE_VARIANTS[vi];
                    auto r = benchmarkVariant(variant, shape, M, gpu_weights);
                    bw_results[si * num_m * num_variants + mi * num_variants + vi] = std::move(r);
                    ++completed;

                    const auto &res = bw_results[si * num_m * num_variants + mi * num_variants + vi];
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

            // Header: Shape | RowWise | V3/MT16 | V3/MT32 | V3/MT64 | V7/MT16 | V7/MT32 | V7/MT64 | Auto | Best | vs RW
            table << fort::header << "Shape" << "Cat" << "N" << "K"
                  << "RowWise";
            for (const auto &v : BLOCKWISE_VARIANTS)
                table << v.name;
            table << "Best BW" << "vs RW" << "Cos" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::left);
            const int total_cols = 4 + 1 + static_cast<int>(num_variants) + 3;
            for (int c = 2; c < total_cols; ++c)
                table.column(static_cast<unsigned>(c)).set_cell_text_align(fort::text_align::right);

            for (size_t si = 0; si < num_shapes; ++si)
            {
                const auto &shape = GEMM_SHAPES[si];
                const auto &rw = rw_results[si * num_m + mi];

                table << shape.name << shape.category
                      << std::to_string(shape.N) << std::to_string(shape.K);

                // Row-wise baseline time
                char buf[32];
                snprintf(buf, sizeof(buf), "%.0f", rw.min_us);
                table << buf;

                // Find best blockwise variant
                int best_idx = 0;
                double best_us = 1e18;
                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    const auto &r = bw_results[si * num_m * num_variants + mi * num_variants + vi];
                    snprintf(buf, sizeof(buf), "%.0f", r.min_us);
                    table << buf;

                    if (r.min_us > 0 && r.correctness_pass && r.min_us < best_us)
                    {
                        best_us = r.min_us;
                        best_idx = static_cast<int>(vi);
                    }
                }

                const auto &best = bw_results[si * num_m * num_variants + mi * num_variants + best_idx];
                double vs_rw = (rw.min_us > 0 && best.min_us > 0) ? rw.min_us / best.min_us : 0.0;

                table << BLOCKWISE_VARIANTS[best_idx].name;
                snprintf(buf, sizeof(buf), "%.2fx", vs_rw);
                table << buf;
                snprintf(buf, sizeof(buf), "%.4f", best.cosine_sim);
                table << buf;
                table << fort::endr;
            }

            char title[128];
            snprintf(title, sizeof(title),
                     "Blockwise VNNI GEMM Sweep — M=%d (%d warmup + %d runs, μs min)",
                     M, WARMUP_RUNS, BENCH_RUNS);
            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }

        // =====================================================================
        // Summary: Best blockwise variant per shape class, averaged across M
        // =====================================================================
        {
            fprintf(stderr, "\n");
            fort::utf8_table summary;
            summary.set_border_style(FT_DOUBLE2_STYLE);
            summary << fort::header << "Shape" << "Category"
                    << "Best Variant" << "Avg BW μs" << "Avg RW μs"
                    << "Ratio (RW/BW)" << "Avg Cosine" << "Gate" << fort::endr;
            summary.column(0).set_cell_text_align(fort::text_align::left);
            summary.column(1).set_cell_text_align(fort::text_align::left);
            for (int c = 2; c <= 7; ++c)
                summary.column(static_cast<unsigned>(c)).set_cell_text_align(fort::text_align::right);

            int pass_count = 0;
            for (size_t si = 0; si < num_shapes; ++si)
            {
                const auto &shape = GEMM_SHAPES[si];

                // For each variant, average min_us across M values
                std::vector<double> variant_avg_us(num_variants, 0.0);
                std::vector<double> variant_avg_cos(num_variants, 0.0);
                std::vector<bool> variant_all_correct(num_variants, true);
                double rw_avg_us = 0.0;

                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    for (size_t mi = 0; mi < num_m; ++mi)
                    {
                        const auto &r = bw_results[si * num_m * num_variants + mi * num_variants + vi];
                        variant_avg_us[vi] += r.min_us;
                        variant_avg_cos[vi] += r.cosine_sim;
                        if (!r.correctness_pass)
                            variant_all_correct[vi] = false;
                    }
                    variant_avg_us[vi] /= static_cast<double>(num_m);
                    variant_avg_cos[vi] /= static_cast<double>(num_m);
                }

                for (size_t mi = 0; mi < num_m; ++mi)
                    rw_avg_us += rw_results[si * num_m + mi].min_us;
                rw_avg_us /= static_cast<double>(num_m);

                // Find best correct variant
                int best_vi = -1;
                double best_avg = 1e18;
                for (size_t vi = 0; vi < num_variants; ++vi)
                {
                    if (variant_all_correct[vi] && variant_avg_us[vi] > 0 && variant_avg_us[vi] < best_avg)
                    {
                        best_avg = variant_avg_us[vi];
                        best_vi = static_cast<int>(vi);
                    }
                }

                if (best_vi < 0)
                    best_vi = 0;

                double ratio = (rw_avg_us > 0 && best_avg > 0) ? rw_avg_us / best_avg : 0.0;
                bool shape_pass = variant_all_correct[best_vi] && ratio > 0;
                if (shape_pass)
                    ++pass_count;

                char bw_buf[16], rw_buf[16], rat_buf[16], cos_buf[16];
                snprintf(bw_buf, sizeof(bw_buf), "%.0f", best_avg);
                snprintf(rw_buf, sizeof(rw_buf), "%.0f", rw_avg_us);
                snprintf(rat_buf, sizeof(rat_buf), "%.2fx", ratio);
                snprintf(cos_buf, sizeof(cos_buf), "%.4f", variant_avg_cos[best_vi]);

                summary << shape.name << shape.category
                        << BLOCKWISE_VARIANTS[best_vi].name
                        << bw_buf << rw_buf << rat_buf << cos_buf
                        << (shape_pass ? "PASS ✓" : "FAIL ✗")
                        << fort::endr;

                // Validate correctness
                EXPECT_TRUE(variant_all_correct[best_vi])
                    << shape.name << " best variant " << BLOCKWISE_VARIANTS[best_vi].name
                    << " failed correctness gate";
            }

            summary << fort::separator;
            char sum_buf[64];
            snprintf(sum_buf, sizeof(sum_buf), "%d/%zu passed", pass_count, num_shapes);
            summary << "" << sum_buf << "" << "" << "" << "" << "" << "" << fort::endr;

            fprintf(stderr, "BEST VARIANT PER SHAPE (averaged across M={128,256,512})\n%s\n",
                    summary.to_string().c_str());
            fprintf(stderr, "Ratio = RowWise_time / Blockwise_time (>1 = blockwise slower)\n");
            fprintf(stderr, "Correctness gate: cosine >= %.4f\n", COSINE_SIM_GATE);
        }

        // =====================================================================
        // Detailed: per-shape × per-M best variant recommendation
        // =====================================================================
        {
            fprintf(stderr, "\n");
            fort::utf8_table detail;
            detail.set_border_style(FT_DOUBLE2_STYLE);
            detail << fort::header << "Shape" << "M" << "Best Variant"
                   << "BW μs" << "RW μs" << "Ratio" << "GFLOPS" << "Cosine" << fort::endr;
            detail.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 7; ++c)
                detail.column(static_cast<unsigned>(c)).set_cell_text_align(fort::text_align::right);

            for (size_t si = 0; si < num_shapes; ++si)
            {
                const auto &shape = GEMM_SHAPES[si];
                for (size_t mi = 0; mi < num_m; ++mi)
                {
                    int M = M_VALUES[mi];
                    const auto &rw = rw_results[si * num_m + mi];

                    int best_vi = 0;
                    double best_us = 1e18;
                    for (size_t vi = 0; vi < num_variants; ++vi)
                    {
                        const auto &r = bw_results[si * num_m * num_variants + mi * num_variants + vi];
                        if (r.min_us > 0 && r.correctness_pass && r.min_us < best_us)
                        {
                            best_us = r.min_us;
                            best_vi = static_cast<int>(vi);
                        }
                    }

                    const auto &best = bw_results[si * num_m * num_variants + mi * num_variants + best_vi];
                    double ratio = (rw.min_us > 0 && best.min_us > 0) ? rw.min_us / best.min_us : 0.0;

                    char bw_buf[16], rw_buf[16], rat_buf[16], gf_buf[16], cos_buf[16];
                    snprintf(bw_buf, sizeof(bw_buf), "%.0f", best.min_us);
                    snprintf(rw_buf, sizeof(rw_buf), "%.0f", rw.min_us);
                    snprintf(rat_buf, sizeof(rat_buf), "%.2fx", ratio);
                    snprintf(gf_buf, sizeof(gf_buf), "%.0f", best.gflops);
                    snprintf(cos_buf, sizeof(cos_buf), "%.4f", best.cosine_sim);

                    detail << shape.name << std::to_string(M)
                           << BLOCKWISE_VARIANTS[best_vi].name
                           << bw_buf << rw_buf << rat_buf << gf_buf << cos_buf
                           << fort::endr;
                }
            }

            fprintf(stderr, "DETAILED: Best blockwise variant per shape × M\n%s\n",
                    detail.to_string().c_str());
        }
#endif
    }

} // anonymous namespace
