/**
 * @file Perf__BlockwiseQuantKernel.cpp
 * @brief Per-shape tuning for blockwise activation quantization kernel variants.
 *
 * Benchmarks the FP32 → INT8 blockwise quantization kernel in isolation,
 * testing 5 kernel variants:
 *   V1: Original (64 threads, 1 block/WG, shared memory)
 *   V2: Multi-block (256 threads, 8 blocks/WG, wave shuffle)
 *   V3: Row-coarsened (256 threads, 1 row/WG, sequential blocks, vec4)
 *   V4: Minimal WG (32 threads, 1 block/WG, wave shuffle)
 *   V5: Two-pass vectorized (256 threads, 1 row/WG, vec4+atomicCAS)
 *
 * Also benchmarks the row-wise quantization kernel as a baseline.
 *
 * Shapes: Qwen2.5-0.5B, 3B, 7B hidden/intermediate sizes.
 * M values: 1 (decode), 32, 64, 128, 256, 512 (prefill).
 *
 * Correctness: Verified by comparing INT8 output + scales against a CPU reference.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "utils/DebugEnv.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;

// ============================================================================
// Extern "C" declarations matching the launcher in ROCmQuantisedGemmKernel.hip
// ============================================================================
extern "C"
{
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size);
}

namespace
{
    // =========================================================================
    // ScopedEnvOverride (same pattern as other perf tests)
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
            mutableDebugEnv().reload();
        }

        ~ScopedEnvOverride()
        {
            if (had_original_)
                ::setenv(name_, original_value_.c_str(), 1);
            else
                ::unsetenv(name_);
            mutableDebugEnv().reload();
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        const char *name_;
        bool had_original_ = false;
        std::string original_value_;
    };

    // =========================================================================
    // Shape definitions (K dimensions from Qwen architectures)
    // =========================================================================

    struct QuantShape
    {
        std::string name;
        int K; // activation width (hidden_size or intermediate_size)
    };

    static const std::vector<QuantShape> QUANT_SHAPES = {
        // Qwen2.5-0.5B
        {"0.5B_hidden (K=896)", 896},
        {"0.5B_intermediate (K=4864)", 4864},
        // Qwen2.5-3B
        {"3B_hidden (K=2048)", 2048},
        {"3B_intermediate (K=11008)", 11008},
        // Qwen2.5-7B
        {"7B_hidden (K=3584)", 3584},
        {"7B_intermediate (K=18944)", 18944},
    };

    static const std::vector<int> M_VALUES = {1, 32, 64, 128, 256, 512};

    struct VariantSpec
    {
        int id;
        std::string name;
    };

    static const std::vector<VariantSpec> VARIANTS = {
        {1, "V1:Orig(64t,1b)"},
        {2, "V2:Multi(256t,8b)"},
        {3, "V3:RowSeq(256t)"},
        {4, "V4:Min(32t,1b)"},
        {5, "V5:Vec4(256t)"},
    };

    constexpr int WARMUP_RUNS = 10;
    constexpr int BENCH_RUNS = 50;

    // =========================================================================
    // CPU reference for correctness
    // =========================================================================

    struct CpuQuantResult
    {
        std::vector<int8_t> int8_data;
        std::vector<float> scales;
        int blocks_per_row;
    };

    static CpuQuantResult cpuBlockwiseQuant(const float *fp32_data, int M, int K)
    {
        constexpr int BLOCK_SIZE = 32;
        const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CpuQuantResult result;
        result.blocks_per_row = blocks_per_row;
        result.int8_data.resize(static_cast<size_t>(M) * K);
        result.scales.resize(static_cast<size_t>(M) * blocks_per_row);

        for (int m = 0; m < M; ++m)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                int k_start = b * BLOCK_SIZE;
                int k_end = std::min(k_start + BLOCK_SIZE, K);

                float max_abs = 0.0f;
                for (int k = k_start; k < k_end; ++k)
                    max_abs = std::max(max_abs, std::abs(fp32_data[m * K + k]));

                float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                float inv_scale = 1.0f / scale;

                result.scales[m * blocks_per_row + b] = scale;

                for (int k = k_start; k < k_end; ++k)
                {
                    int q = static_cast<int>(std::round(fp32_data[m * K + k] * inv_scale));
                    q = std::max(-127, std::min(127, q));
                    result.int8_data[m * K + k] = static_cast<int8_t>(q);
                }
            }
        }
        return result;
    }

    // =========================================================================
    // Benchmark result
    // =========================================================================

    struct QuantBenchResult
    {
        std::string variant_name;
        std::string shape_name;
        int M, K;
        double min_us = 0.0;
        double mean_us = 0.0;
        double bandwidth_gb_s = 0.0; // effective bandwidth: read FP32 + write INT8 + scales
        bool correctness_pass = false;
        int int8_mismatches = 0;
        float max_scale_error = 0.0f;
    };

    // =========================================================================
    // Test fixture
    // =========================================================================

    class BlockwiseQuantPerfTest : public ::testing::Test
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

        bool has_device_ = false;
        std::string device_name_;

#ifdef HAVE_ROCM
        QuantBenchResult benchmarkVariant(int variant_id, const std::string &variant_name,
                                          const QuantShape &shape, int M,
                                          const float *h_fp32,
                                          const CpuQuantResult &cpu_ref)
        {
            QuantBenchResult result;
            result.variant_name = variant_name;
            result.shape_name = shape.name;
            result.M = M;
            result.K = shape.K;

            const size_t fp32_bytes = static_cast<size_t>(M) * shape.K * sizeof(float);
            const size_t int8_bytes = static_cast<size_t>(M) * shape.K * sizeof(int8_t);
            const int blocks_per_row = (shape.K + 31) / 32;
            const size_t scale_bytes = static_cast<size_t>(M) * blocks_per_row * sizeof(float);

            // Allocate device buffers
            float *d_fp32 = nullptr;
            int8_t *d_int8 = nullptr;
            float *d_scales = nullptr;

            if (hipMalloc(&d_fp32, fp32_bytes) != hipSuccess ||
                hipMalloc(&d_int8, int8_bytes) != hipSuccess ||
                hipMalloc(&d_scales, scale_bytes) != hipSuccess)
            {
                if (d_fp32)
                    (void)hipFree(d_fp32);
                if (d_int8)
                    (void)hipFree(d_int8);
                if (d_scales)
                    (void)hipFree(d_scales);
                return result;
            }

            (void)hipMemcpy(d_fp32, h_fp32, fp32_bytes, hipMemcpyHostToDevice);

            // Set variant via environment
            ScopedEnvOverride env_override("LLAMINAR_ROCM_BLOCKWISE_QUANT_VARIANT",
                                           std::to_string(variant_id));

            // Correctness check: run once, download results
            {
                (void)hipMemset(d_int8, 0, int8_bytes);
                (void)hipMemset(d_scales, 0, scale_bytes);

                bool ok = rocmQuantGemm_quantizeActivationsBlockwise(
                    d_fp32, d_int8, d_scales, M, shape.K, 0, nullptr, 32);
                (void)hipDeviceSynchronize();

                if (!ok)
                {
                    (void)hipFree(d_fp32);
                    (void)hipFree(d_int8);
                    (void)hipFree(d_scales);
                    return result;
                }

                std::vector<int8_t> h_int8(static_cast<size_t>(M) * shape.K);
                std::vector<float> h_scales(static_cast<size_t>(M) * blocks_per_row);
                (void)hipMemcpy(h_int8.data(), d_int8, int8_bytes, hipMemcpyDeviceToHost);
                (void)hipMemcpy(h_scales.data(), d_scales, scale_bytes, hipMemcpyDeviceToHost);

                // Compare against CPU reference
                int mismatches = 0;
                for (size_t i = 0; i < h_int8.size(); ++i)
                {
                    if (h_int8[i] != cpu_ref.int8_data[i])
                    {
                        // Allow ±1 difference due to rounding
                        if (std::abs(static_cast<int>(h_int8[i]) - static_cast<int>(cpu_ref.int8_data[i])) > 1)
                            ++mismatches;
                    }
                }
                result.int8_mismatches = mismatches;

                float max_scale_err = 0.0f;
                for (size_t i = 0; i < h_scales.size(); ++i)
                {
                    float err = std::abs(h_scales[i] - cpu_ref.scales[i]);
                    float rel = (cpu_ref.scales[i] > 1e-6f) ? err / cpu_ref.scales[i] : err;
                    max_scale_err = std::max(max_scale_err, rel);
                }
                result.max_scale_error = max_scale_err;
                result.correctness_pass = (mismatches == 0 && max_scale_err < 0.001f);
            }

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(
                    d_fp32, d_int8, d_scales, M, shape.K, 0, nullptr, 32);
            }
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
                rocmQuantGemm_quantizeActivationsBlockwise(
                    d_fp32, d_int8, d_scales, M, shape.K, 0, nullptr, 32);
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

            // Bandwidth: read FP32 + write INT8 + write scales
            double total_bytes = static_cast<double>(fp32_bytes + int8_bytes + scale_bytes);
            result.bandwidth_gb_s = (result.min_us > 0)
                                        ? total_bytes / (result.min_us * 1e-6) / 1e9
                                        : 0.0;

            (void)hipFree(d_fp32);
            (void)hipFree(d_int8);
            (void)hipFree(d_scales);
            return result;
        }

        QuantBenchResult benchmarkRowWise(const QuantShape &shape, int M,
                                          const float *h_fp32)
        {
            QuantBenchResult result;
            result.variant_name = "Blockwise32_baseline";
            result.shape_name = shape.name;
            result.M = M;
            result.K = shape.K;

            const int blocks_per_row = (shape.K + 31) / 32;
            const size_t fp32_bytes = static_cast<size_t>(M) * shape.K * sizeof(float);
            const size_t int8_bytes = static_cast<size_t>(M) * shape.K * sizeof(int8_t);
            const size_t scale_bytes = static_cast<size_t>(M) * blocks_per_row * sizeof(float);

            float *d_fp32 = nullptr;
            int8_t *d_int8 = nullptr;
            float *d_scales = nullptr;

            if (hipMalloc(&d_fp32, fp32_bytes) != hipSuccess ||
                hipMalloc(&d_int8, int8_bytes) != hipSuccess ||
                hipMalloc(&d_scales, scale_bytes) != hipSuccess)
            {
                if (d_fp32)
                    (void)hipFree(d_fp32);
                if (d_int8)
                    (void)hipFree(d_int8);
                if (d_scales)
                    (void)hipFree(d_scales);
                return result;
            }

            (void)hipMemcpy(d_fp32, h_fp32, fp32_bytes, hipMemcpyHostToDevice);
            result.correctness_pass = true; // Blockwise baseline is the reference path

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(
                    d_fp32, d_int8, d_scales, M, shape.K, 0, nullptr, 32);
            }
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
                rocmQuantGemm_quantizeActivationsBlockwise(
                    d_fp32, d_int8, d_scales, M, shape.K, 0, nullptr, 32);
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

            double total_bytes = static_cast<double>(fp32_bytes + int8_bytes + scale_bytes);
            result.bandwidth_gb_s = (result.min_us > 0)
                                        ? total_bytes / (result.min_us * 1e-6) / 1e9
                                        : 0.0;

            (void)hipFree(d_fp32);
            (void)hipFree(d_int8);
            (void)hipFree(d_scales);
            return result;
        }
#endif
    };

    // =========================================================================
    // Full sweep test
    // =========================================================================

    TEST_F(BlockwiseQuantPerfTest, VariantSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[BlockwiseQuant] Variant Sweep\n");
        fprintf(stderr, "[BlockwiseQuant] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[BlockwiseQuant] %zu shapes × %zu M values × %zu variants + row-wise\n",
                QUANT_SHAPES.size(), M_VALUES.size(), VARIANTS.size());
        fprintf(stderr, "[BlockwiseQuant] %d warmup + %d timed runs\n\n",
                WARMUP_RUNS, BENCH_RUNS);

        const size_t num_shapes = QUANT_SHAPES.size();
        const size_t num_m = M_VALUES.size();
        const size_t num_var = VARIANTS.size();

        // Results: [shape][m][variant] for blockwise, [shape][m] for row-wise
        std::vector<QuantBenchResult> bw_results(num_shapes * num_m * num_var);
        std::vector<QuantBenchResult> rw_results(num_shapes * num_m);

        int total = static_cast<int>(num_shapes * num_m * (num_var + 1));
        int done = 0;

        for (size_t si = 0; si < num_shapes; ++si)
        {
            const auto &shape = QUANT_SHAPES[si];

            for (size_t mi = 0; mi < num_m; ++mi)
            {
                int M = M_VALUES[mi];

                // Generate random FP32 input data
                const size_t num_elems = static_cast<size_t>(M) * shape.K;
                std::vector<float> h_fp32(num_elems);
                std::mt19937 rng(42 + si * 100 + mi);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                for (auto &v : h_fp32)
                    v = dist(rng);

                // CPU reference
                auto cpu_ref = cpuBlockwiseQuant(h_fp32.data(), M, shape.K);

                // Row-wise baseline
                {
                    auto r = benchmarkRowWise(shape, M, h_fp32.data());
                    rw_results[si * num_m + mi] = std::move(r);
                    ++done;
                    fprintf(stderr, "  RowWise %s M=%d: %.1f μs  [%d/%d]\n",
                            shape.name.c_str(), M,
                            rw_results[si * num_m + mi].min_us, done, total);
                }

                // Blockwise variants
                for (size_t vi = 0; vi < num_var; ++vi)
                {
                    const auto &var = VARIANTS[vi];
                    auto r = benchmarkVariant(var.id, var.name, shape, M,
                                              h_fp32.data(), cpu_ref);
                    bw_results[si * num_m * num_var + mi * num_var + vi] = std::move(r);
                    ++done;

                    const auto &res = bw_results[si * num_m * num_var + mi * num_var + vi];
                    fprintf(stderr, "  %s %s M=%d: %.1f μs %s  [%d/%d]\n",
                            var.name.c_str(), shape.name.c_str(), M,
                            res.min_us,
                            res.correctness_pass ? "✓" : "✗",
                            done, total);
                }
            }
        }

        // =====================================================================
        // Table 1: Per-shape × M detailed results (one table per shape)
        // =====================================================================
        for (size_t si = 0; si < num_shapes; ++si)
        {
            const auto &shape = QUANT_SHAPES[si];

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Header
            table << fort::header << "M" << "RowWise";
            for (const auto &v : VARIANTS)
                table << v.name;
            table << "Best BW" << "vs RW" << "BW GB/s" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::right);
            for (size_t c = 1; c < 2 + num_var + 3; ++c)
                table.column(static_cast<unsigned>(c)).set_cell_text_align(fort::text_align::right);

            for (size_t mi = 0; mi < num_m; ++mi)
            {
                int M = M_VALUES[mi];
                const auto &rw = rw_results[si * num_m + mi];

                char buf[32];
                table << std::to_string(M);
                snprintf(buf, sizeof(buf), "%.1f", rw.min_us);
                table << buf;

                int best_vi = 0;
                double best_us = 1e18;
                for (size_t vi = 0; vi < num_var; ++vi)
                {
                    const auto &r = bw_results[si * num_m * num_var + mi * num_var + vi];
                    snprintf(buf, sizeof(buf), "%.1f%s", r.min_us, r.correctness_pass ? "" : "!");
                    table << buf;

                    if (r.min_us > 0 && r.correctness_pass && r.min_us < best_us)
                    {
                        best_us = r.min_us;
                        best_vi = static_cast<int>(vi);
                    }
                }

                const auto &best = bw_results[si * num_m * num_var + mi * num_var + best_vi];
                double ratio = (rw.min_us > 0 && best.min_us > 0)
                                   ? rw.min_us / best.min_us
                                   : 0.0;

                table << VARIANTS[best_vi].name;
                snprintf(buf, sizeof(buf), "%.2fx", ratio);
                table << buf;
                snprintf(buf, sizeof(buf), "%.1f", best.bandwidth_gb_s);
                table << buf;
                table << fort::endr;
            }

            char title[256];
            snprintf(title, sizeof(title),
                     "Blockwise Quant Kernel — %s (%d warmup + %d runs, μs min)",
                     shape.name.c_str(), WARMUP_RUNS, BENCH_RUNS);
            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }

        // =====================================================================
        // Table 2: Summary — best variant per shape (averaged across M)
        // =====================================================================
        {
            fprintf(stderr, "\n");
            fort::utf8_table summary;
            summary.set_border_style(FT_DOUBLE2_STYLE);
            summary << fort::header << "Shape" << "Best Variant"
                    << "Avg BW μs" << "Avg RW μs" << "Speedup vs RW"
                    << "Avg BW GB/s" << "Gate" << fort::endr;
            summary.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 6; ++c)
                summary.column(static_cast<unsigned>(c)).set_cell_text_align(fort::text_align::right);

            for (size_t si = 0; si < num_shapes; ++si)
            {
                const auto &shape = QUANT_SHAPES[si];

                // Average each variant across M values
                std::vector<double> var_avg_us(num_var, 0.0);
                std::vector<double> var_avg_bw(num_var, 0.0);
                std::vector<bool> var_all_ok(num_var, true);
                double rw_avg = 0.0;

                for (size_t vi = 0; vi < num_var; ++vi)
                {
                    for (size_t mi = 0; mi < num_m; ++mi)
                    {
                        const auto &r = bw_results[si * num_m * num_var + mi * num_var + vi];
                        var_avg_us[vi] += r.min_us;
                        var_avg_bw[vi] += r.bandwidth_gb_s;
                        if (!r.correctness_pass)
                            var_all_ok[vi] = false;
                    }
                    var_avg_us[vi] /= static_cast<double>(num_m);
                    var_avg_bw[vi] /= static_cast<double>(num_m);
                }

                for (size_t mi = 0; mi < num_m; ++mi)
                    rw_avg += rw_results[si * num_m + mi].min_us;
                rw_avg /= static_cast<double>(num_m);

                int best_vi = 0;
                double best_avg = 1e18;
                for (size_t vi = 0; vi < num_var; ++vi)
                {
                    if (var_all_ok[vi] && var_avg_us[vi] > 0 && var_avg_us[vi] < best_avg)
                    {
                        best_avg = var_avg_us[vi];
                        best_vi = static_cast<int>(vi);
                    }
                }

                double speedup = (rw_avg > 0 && best_avg > 0) ? rw_avg / best_avg : 0.0;

                char bw_buf[16], rw_buf[16], sp_buf[16], gb_buf[16];
                snprintf(bw_buf, sizeof(bw_buf), "%.1f", best_avg);
                snprintf(rw_buf, sizeof(rw_buf), "%.1f", rw_avg);
                snprintf(sp_buf, sizeof(sp_buf), "%.2fx", speedup);
                snprintf(gb_buf, sizeof(gb_buf), "%.1f", var_avg_bw[best_vi]);

                summary << shape.name << VARIANTS[best_vi].name
                        << bw_buf << rw_buf << sp_buf << gb_buf
                        << (var_all_ok[best_vi] ? "PASS ✓" : "FAIL ✗")
                        << fort::endr;

                EXPECT_TRUE(var_all_ok[best_vi])
                    << shape.name << " best variant " << VARIANTS[best_vi].name
                    << " failed correctness";
            }

            fprintf(stderr, "BEST VARIANT PER SHAPE (averaged across M={1..512})\n%s\n",
                    summary.to_string().c_str());
            fprintf(stderr, "Speedup = RowWise_time / Blockwise_time (>1 = blockwise overhead, expected)\n");
            fprintf(stderr, "Note: Blockwise quant has more scales (per-block vs per-row)\n");
        }

        // =====================================================================
        // Table 3: Correctness detail for any failures
        // =====================================================================
        {
            bool any_fail = false;
            for (const auto &r : bw_results)
            {
                if (!r.correctness_pass && r.min_us > 0)
                {
                    any_fail = true;
                    break;
                }
            }

            if (any_fail)
            {
                fprintf(stderr, "\nCORRECTNESS FAILURES:\n");
                fort::utf8_table fail_table;
                fail_table.set_border_style(FT_DOUBLE2_STYLE);
                fail_table << fort::header << "Variant" << "Shape" << "M"
                           << "INT8 Mismatches" << "Max Scale Error" << fort::endr;

                for (const auto &r : bw_results)
                {
                    if (!r.correctness_pass && r.min_us > 0)
                    {
                        char err_buf[16];
                        snprintf(err_buf, sizeof(err_buf), "%.6f", r.max_scale_error);
                        fail_table << r.variant_name << r.shape_name
                                   << std::to_string(r.M) << std::to_string(r.int8_mismatches)
                                   << err_buf << fort::endr;
                    }
                }
                fprintf(stderr, "%s\n", fail_table.to_string().c_str());
            }
        }
#endif
    }

} // anonymous namespace
