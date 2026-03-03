/**
 * @file Perf__NativeVNNI_GEMM.cpp
 * @brief Performance and correctness benchmark for native-VNNI GEMM kernels
 *
 * Measures M>1 (prefill) throughput for Q4_0 and IQ4_NL formats using the
 * native-VNNI GEMM kernel, comparing against the INT8 VNNI GEMM (V3/V7)
 * baseline on identical shapes.
 *
 * Benchmarked dimensions are drawn from Qwen2.5-0.5B, 3B, and 7B layer
 * shapes at multiple sequence lengths (M=32, 64, 128, 256).
 *
 * Metrics reported:
 *   - Kernel time (μs): min/mean across benchmark runs
 *   - Speedup vs INT8: int8_min_us / native_vnni_min_us (>1 = faster)
 *   - Theoretical speedup: 8.0 / bpw (from streaming fewer weight bytes)
 *   - Kernel efficiency: actual_speedup / theoretical_speedup × 100%
 *   - Cosine similarity: output vs FP32 reference (correctness gate: >0.9999)
 *
 * The benchmark uses multiply_tensor() which includes:
 *   1. FP32→INT8 activation quantization on GPU
 *   2. Native-VNNI GEMM kernel dispatch (Step 1b) or INT8 GEMM (V3/V7)
 *   3. Per-block FP16 scale application (native-VNNI) or scaling epilogue (INT8)
 *
 * @note Requires ROCm device. Tests skip if no GPU is available.
 * @note Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "kernels/rocm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{

    // =============================================================================
    // Constants
    // =============================================================================

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    /// Correctness gate: cosine similarity between native-VNNI and FP32 reference
    constexpr float COSINE_SIM_GATE = 0.9999f;

    /// Performance gate: speedup over INT8 GEMM baseline
    constexpr float SPEEDUP_GATE = 1.8f;

    // =============================================================================
    // Format descriptors (sprint: Q4_0 and IQ4_NL only)
    // =============================================================================

    struct GEMMFormatSpec
    {
        std::string name;
        double bpw;

        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    static const std::vector<GEMMFormatSpec> GEMM_FORMATS = {
        {"Q4_0", 4.5, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", 4.5, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
    };

    // =============================================================================
    // GEMM shape definitions (N×K with variable M)
    // =============================================================================

    struct GEMMShape
    {
        std::string name;
        int N;
        int K;
    };

    /// Sequence lengths to benchmark (typical prefill sizes)
    static const std::vector<int> M_VALUES = {32, 64, 128, 256};

    // Qwen2.5-0.5B:  hidden=896,  intermediate=4864
    // Qwen2.5-3B:    hidden=2048, intermediate=11008
    // Qwen2.5-7B:    hidden=3584, intermediate=18944
    static const std::vector<GEMMShape> GEMM_SHAPES = {
        // Qwen2.5-0.5B
        {"0.5B_AttnOut", 896, 896},
        {"0.5B_FFN_Up", 4864, 896},
        {"0.5B_FFN_Dn", 896, 4864},
        {"0.5B_LM_Head", 151936, 896},
        // Qwen2.5-3B
        {"3B_AttnOut", 2048, 2048},
        {"3B_FFN_Up", 11008, 2048},
        {"3B_FFN_Dn", 2048, 11008},
        {"3B_LM_Head", 151936, 2048},
        // Qwen2.5-7B
        {"7B_AttnOut", 3584, 3584},
        {"7B_FFN_Up", 18944, 3584},
        {"7B_FFN_Dn", 3584, 18944},
    };

    // =============================================================================
    // Benchmark result
    // =============================================================================

    struct GEMMBenchResult
    {
        std::string format_name;
        double bpw;
        std::string shape_name;
        int M, N, K;

        double min_us;
        double mean_us;
        double stddev_us;

        double native_weight_bytes; // Native-VNNI payload + scales bytes read

        // INT8 reference comparison
        double int8_min_us = 0.0;
        double speedup_vs_int8 = 0.0;     // int8_min_us / min_us
        double theoretical_speedup = 0.0; // 8.0 / bpw
        double kernel_efficiency = 0.0;   // (speedup / theoretical) × 100

        // Correctness
        float cosine_sim = 0.0f;
        bool correctness_pass = false;

        // GFLOPS
        double gflops = 0.0;
    };

    // =============================================================================
    // Statistics helper
    // =============================================================================

    static void computeStats(const std::vector<double> &times_us,
                             double &mean, double &min_val,
                             double &max_val, double &stddev)
    {
        mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) /
               static_cast<double>(times_us.size());
        min_val = *std::min_element(times_us.begin(), times_us.end());
        max_val = *std::max_element(times_us.begin(), times_us.end());

        double sq_sum = 0.0;
        for (double t : times_us)
            sq_sum += (t - mean) * (t - mean);
        stddev = std::sqrt(sq_sum / static_cast<double>(times_us.size()));
    }

    // =============================================================================
    // FP32 reference GEMM (CPU, for correctness validation)
    //
    // Computes C[M×N] = A[M×K] × B^T[N×K]  (B stored as [N×K] row-major)
    // OpenMP-parallelized across M×N output elements.
    // =============================================================================

    static void referenceFP32Gemm(const float *A, const float *B_dequant,
                                  float *C, int M, int N, int K)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float acc = 0.0f;
                for (int ki = 0; ki < K; ++ki)
                    acc += A[m * K + ki] * B_dequant[n * K + ki];
                C[m * N + n] = acc;
            }
        }
    }

    // =============================================================================
    // Parallel cosine similarity (OpenMP reduction)
    // =============================================================================

    static float parallelCosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
#pragma omp parallel for reduction(+ : dot, norm_a, norm_b) schedule(static)
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return (denom > 0.0) ? static_cast<float>(dot / denom) : 0.0f;
    }

    // =============================================================================
    // Test fixture
    // =============================================================================

    class NativeVNNIGEMMPerfTest : public ::testing::Test
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

        bool has_device_ = false;
        std::string device_name_;

#ifdef HAVE_ROCM

        /// Time a GEMM kernel call. Returns sorted timing vector in μs.
        std::vector<double> timeGEMMKernel(ROCmQuantisedGemmKernel &kernel,
                                           TensorBase *input, TensorBase *output,
                                           int M, int N, int K)
        {
            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input, output, M, N, K);
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
                kernel.multiply_tensor(input, output, M, N, K);
                (void)hipEventRecord(stop);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);

            std::sort(times_us.begin(), times_us.end());
            return times_us;
        }

        /// Benchmark INT8 VNNI GEMM reference (Q8_0 → INT8 V3/V7) for a shape+M.
        /// Returns min kernel time in μs.
        double benchmarkINT8GEMMReference(int M, int N, int K)
        {
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)});
            if (!weights)
                return 0.0;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return 0.0;

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, N, K);
            const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            if (!workspace->allocate(reqs))
                return 0.0;
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});
            if (!input->ensureOnDevice(DeviceId::rocm(0)))
                return 0.0;
            if (!output->allocateOnDevice(DeviceId::rocm(0)))
                return 0.0;

            auto times = timeGEMMKernel(kernel, input.get(), output.get(), M, N, K);
            kernel.unbindWorkspace();

            return times.empty() ? 0.0 : times.front(); // min
        }

        /// Run a single format+shape+M benchmark, including correctness check.
        GEMMBenchResult benchmarkGEMM(const GEMMFormatSpec &fmt,
                                      const GEMMShape &shape, int M,
                                      double int8_ref_us)
        {
            GEMMBenchResult result{};
            result.format_name = fmt.name;
            result.bpw = fmt.bpw;
            result.shape_name = shape.name;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            // 1. Create quantized weights and pack
            auto weights = fmt.create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            EXPECT_NE(weights, nullptr);
            if (!weights)
                return result;

            ROCmPackedWeights packed;
            EXPECT_TRUE(packWeightsToROCm(weights.get(), packed));
            if (packed.native_vnni_payload.empty())
            {
                fprintf(stderr, "  [SKIP] %s: no native-VNNI payload\n",
                        fmt.name.c_str());
                return result;
            }

            // Native-VNNI weight bytes: payload + scales
            result.native_weight_bytes =
                static_cast<double>(packed.native_vnni_payload.size()) +
                static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t));

            // 2. Create kernel + workspace
            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (8 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            EXPECT_TRUE(workspace->allocate(reqs));
            kernel.bindWorkspace(workspace.get());

            // 3. Create input/output tensors and upload
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            EXPECT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
            EXPECT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

            // 4. Correctness: compute FP32 reference on CPU
            {
                // Dequantize weights to FP32 (data() on quantized tensors returns FP32 cache)
                const float *w_fp32 = weights->data();
                EXPECT_NE(w_fp32, nullptr);

                if (w_fp32)
                {
                    // Run one GEMM on GPU to get the output
                    kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                    (void)hipDeviceSynchronize();

                    // Mark output as device-dirty so data() triggers D2H sync
                    output->mark_device_dirty();

                    // Read back GPU output (triggers D2H sync via coherence)
                    auto *out_fp32 = dynamic_cast<FP32Tensor *>(output.get());
                    EXPECT_NE(out_fp32, nullptr);
                    const float *gpu_data = out_fp32->data();

                    // Compute CPU reference
                    const float *a_data = dynamic_cast<const FP32Tensor *>(input.get())->data();
                    std::vector<float> ref_output(static_cast<size_t>(M) * shape.N, 0.0f);
                    referenceFP32Gemm(a_data, w_fp32, ref_output.data(), M, shape.N, shape.K);

                    // Cosine similarity (parallel)
                    result.cosine_sim = parallelCosineSimilarity(
                        gpu_data, ref_output.data(),
                        static_cast<size_t>(M) * shape.N);
                    result.correctness_pass = (result.cosine_sim >= COSINE_SIM_GATE);

                    // Re-upload output for benchmarking (data() call synced D2H)
                    output->ensureOnDevice(DeviceId::rocm(0));
                }
            }

            // 5. Timed runs
            auto times = timeGEMMKernel(kernel, input.get(), output.get(),
                                        M, shape.N, shape.K);

            double max_us;
            computeStats(times, result.mean_us, result.min_us, max_us, result.stddev_us);

            // 6. Compute derived metrics
            double flops = 2.0 * M * static_cast<double>(shape.N) * shape.K;
            result.gflops = flops / (result.min_us * 1e-6) / 1e9;
            result.theoretical_speedup = 8.0 / result.bpw;

            if (int8_ref_us > 0.0 && result.min_us > 0.0)
            {
                result.int8_min_us = int8_ref_us;
                result.speedup_vs_int8 = int8_ref_us / result.min_us;
                result.kernel_efficiency =
                    (result.speedup_vs_int8 / result.theoretical_speedup) * 100.0;
            }

            kernel.unbindWorkspace();
            return result;
        }
#endif
    };

    // =============================================================================
    // Test 1: Correctness validation (Q4_0 and IQ4_NL, all shapes, M=128)
    //
    // Gate: cosine similarity > 0.9999 vs FP32 reference
    // =============================================================================

    TEST_F(NativeVNNIGEMMPerfTest, Correctness_AllFormats_AllShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const int M = 128;

        fprintf(stderr, "\n[NativeVNNI GEMM] Correctness Test (M=%d)\n", M);
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] Gate: cosine similarity >= %.4f\n",
                COSINE_SIM_GATE);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "Shape" << "M" << "N" << "K"
              << "Cosine Sim" << "Status" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c = 2; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        int pass_count = 0, total_count = 0;

        for (const auto &fmt : GEMM_FORMATS)
        {
            for (const auto &shape : GEMM_SHAPES)
            {
                auto r = benchmarkGEMM(fmt, shape, M, 0.0);
                ++total_count;

                char buf_cos[16];
                snprintf(buf_cos, sizeof(buf_cos), "%.6f", r.cosine_sim);

                const char *status = r.correctness_pass ? "PASS ✓" : "FAIL ✗";
                if (r.correctness_pass)
                    ++pass_count;

                table << r.format_name << r.shape_name
                      << std::to_string(M)
                      << std::to_string(r.N) << std::to_string(r.K)
                      << buf_cos << status << fort::endr;

                EXPECT_GE(r.cosine_sim, COSINE_SIM_GATE)
                    << fmt.name << " " << shape.name
                    << " cosine=" << r.cosine_sim;
            }
        }

        table << fort::separator;
        char summary[64];
        snprintf(summary, sizeof(summary), "%d/%d passed", pass_count, total_count);
        table << "" << summary << "" << "" << "" << "" << "" << fort::endr;

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    // =============================================================================
    // Test 2: Performance matrix — all formats × shapes × M values
    //
    // Gate: 1.8x speedup over INT8 GEMM baseline
    // =============================================================================

    TEST_F(NativeVNNIGEMMPerfTest, Performance_AllFormats_AllShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI GEMM] Performance Benchmark\n");
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI GEMM] %zu formats × %zu shapes × %zu M values\n",
                GEMM_FORMATS.size(), GEMM_SHAPES.size(), M_VALUES.size());
        fprintf(stderr, "[NativeVNNI GEMM] %d warmup + %d timed runs each\n",
                WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr, "[NativeVNNI GEMM] Performance gate: %.1fx speedup vs INT8 GEMM\n",
                SPEEDUP_GATE);

        // =====================================================================
        // Phase 1: Benchmark INT8 GEMM reference for each (shape, M)
        // =====================================================================
        fprintf(stderr, "\n[Phase 1] Benchmarking INT8 GEMM reference (V3/V7)...\n");

        // Key: "shape_name:M" → min time in μs
        std::unordered_map<std::string, double> int8_ref;

        for (const auto &shape : GEMM_SHAPES)
        {
            for (int M : M_VALUES)
            {
                std::string key = shape.name + ":" + std::to_string(M);
                double ref_us = benchmarkINT8GEMMReference(M, shape.N, shape.K);
                int8_ref[key] = ref_us;
                fprintf(stderr, "  INT8 %s M=%d: %.1f μs\n",
                        shape.name.c_str(), M, ref_us);
            }
        }

        // =====================================================================
        // Phase 2: Benchmark native-VNNI GEMM for each format
        // =====================================================================
        fprintf(stderr, "\n[Phase 2] Benchmarking native-VNNI GEMM...\n");

        std::vector<GEMMBenchResult> results;
        results.reserve(GEMM_FORMATS.size() * GEMM_SHAPES.size() * M_VALUES.size());

        for (const auto &fmt : GEMM_FORMATS)
        {
            fprintf(stderr, "\n  Format: %s (%.1f bpw)\n", fmt.name.c_str(), fmt.bpw);
            for (const auto &shape : GEMM_SHAPES)
            {
                for (int M : M_VALUES)
                {
                    std::string key = shape.name + ":" + std::to_string(M);
                    double ref_us = int8_ref.count(key) ? int8_ref[key] : 0.0;

                    auto r = benchmarkGEMM(fmt, shape, M, ref_us);
                    fprintf(stderr, "    %s M=%d: %.1f μs (%.2fx vs INT8, cos=%.6f %s)\n",
                            shape.name.c_str(), M, r.min_us,
                            r.speedup_vs_int8, r.cosine_sim,
                            r.correctness_pass ? "✓" : "✗");
                    results.push_back(std::move(r));
                }
            }
        }

        // =====================================================================
        // Phase 3: Per-format summary tables (grouped by M)
        // =====================================================================
        for (const auto &fmt : GEMM_FORMATS)
        {
            for (int M : M_VALUES)
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                char title[256];
                snprintf(title, sizeof(title),
                         "%s (%.1f bpw) | M=%d | %d warmup + %d runs",
                         fmt.name.c_str(), fmt.bpw, M, WARMUP_RUNS, BENCH_RUNS);

                table << fort::header
                      << "Shape" << "N" << "K"
                      << "NVNNI μs" << "INT8 μs" << "Speedup"
                      << "Theoret." << "Kern Eff"
                      << "GFLOPS" << "Cosine" << "Gate" << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (int c = 1; c <= 10; ++c)
                    table.column(c).set_cell_text_align(fort::text_align::right);

                for (const auto &r : results)
                {
                    if (r.format_name != fmt.name || r.M != M)
                        continue;

                    char b_min[16], b_int8[16], b_speedup[16];
                    char b_theo[16], b_keff[16], b_gflops[16];
                    char b_cos[16];

                    snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                    snprintf(b_int8, sizeof(b_int8), "%.1f", r.int8_min_us);
                    snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                    snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                    snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                    snprintf(b_gflops, sizeof(b_gflops), "%.0f", r.gflops);
                    snprintf(b_cos, sizeof(b_cos), "%.6f", r.cosine_sim);

                    const char *gate_str = "";
                    if (r.speedup_vs_int8 >= SPEEDUP_GATE && r.correctness_pass)
                        gate_str = "PASS ✓";
                    else if (!r.correctness_pass)
                        gate_str = "COS ✗";
                    else if (r.speedup_vs_int8 < SPEEDUP_GATE)
                        gate_str = "PERF ✗";

                    table << r.shape_name
                          << std::to_string(r.N) << std::to_string(r.K)
                          << b_min << b_int8 << b_speedup
                          << b_theo << b_keff
                          << b_gflops << b_cos << gate_str << fort::endr;
                }

                fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
            }
        }

        // =====================================================================
        // Phase 4: Grand summary — average across all shapes per format per M
        // =====================================================================
        fprintf(stderr, "\n");
        fort::utf8_table grand;
        grand.set_border_style(FT_DOUBLE2_STYLE);
        grand << fort::header
              << "Format" << "BPW" << "M"
              << "Avg NVNNI μs" << "Avg INT8 μs" << "Avg Speedup"
              << "Theoretical" << "Avg Kern Eff" << "Avg Cosine"
              << fort::endr;

        grand.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            grand.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : GEMM_FORMATS)
        {
            for (int M : M_VALUES)
            {
                double tot_min = 0, tot_int8 = 0, tot_speedup = 0;
                double tot_keff = 0, tot_cos = 0;
                int count = 0;

                for (const auto &r : results)
                {
                    if (r.format_name != fmt.name || r.M != M)
                        continue;
                    tot_min += r.min_us;
                    tot_int8 += r.int8_min_us;
                    tot_speedup += r.speedup_vs_int8;
                    tot_keff += r.kernel_efficiency;
                    tot_cos += r.cosine_sim;
                    ++count;
                }
                if (count == 0)
                    continue;

                char b_bpw[8], b_m[8], b_min[16], b_int8[16];
                char b_speedup[16], b_theo[16], b_keff[16], b_cos[16];

                snprintf(b_bpw, sizeof(b_bpw), "%.1f", fmt.bpw);
                snprintf(b_m, sizeof(b_m), "%d", M);
                snprintf(b_min, sizeof(b_min), "%.1f", tot_min / count);
                snprintf(b_int8, sizeof(b_int8), "%.1f", tot_int8 / count);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", tot_speedup / count);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", 8.0 / fmt.bpw);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", tot_keff / count);
                snprintf(b_cos, sizeof(b_cos), "%.6f", tot_cos / count);

                grand << fmt.name << b_bpw << b_m
                      << b_min << b_int8 << b_speedup
                      << b_theo << b_keff << b_cos << fort::endr;
            }
        }

        fprintf(stderr,
                "GRAND SUMMARY: NativeVNNI GEMM vs INT8 GEMM (V3/V7)\n%s\n",
                grand.to_string().c_str());
        fprintf(stderr, "Speedup = INT8_time / NativeVNNI_time (>1x = faster)\n");
        fprintf(stderr, "Theoretical = 8.0/BPW = %.2fx (bandwidth-optimal for 4.5 bpw)\n",
                8.0 / 4.5);
        fprintf(stderr, "Performance gate: %.1fx | Correctness gate: cosine >= %.4f\n",
                SPEEDUP_GATE, COSINE_SIM_GATE);
#endif
    }

    // =============================================================================
    // Test 3: Focused benchmark — single shape, all M values, both formats
    //
    // Quick iteration test for kernel tuning on a representative shape.
    // =============================================================================

    TEST_F(NativeVNNIGEMMPerfTest, Focused_3B_FFN_Up)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const GEMMShape shape{"3B_FFN_Up", 11008, 2048};

        fprintf(stderr, "\n[NativeVNNI GEMM] Focused: %s (N=%d K=%d)\n",
                shape.name.c_str(), shape.N, shape.K);
        fprintf(stderr, "[NativeVNNI GEMM] Device: %s\n", device_name_.c_str());

        // INT8 references for each M
        std::unordered_map<int, double> int8_refs;
        for (int M : M_VALUES)
        {
            int8_refs[M] = benchmarkINT8GEMMReference(M, shape.N, shape.K);
            fprintf(stderr, "  INT8 ref M=%d: %.1f μs\n", M, int8_refs[M]);
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "M"
              << "NVNNI μs" << "INT8 μs" << "Speedup"
              << "Theoret." << "Kern Eff"
              << "GFLOPS" << "Cosine" << "Gate" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : GEMM_FORMATS)
        {
            for (int M : M_VALUES)
            {
                auto r = benchmarkGEMM(fmt, shape, M, int8_refs[M]);

                char b_m[8], b_min[16], b_int8[16], b_speedup[16];
                char b_theo[16], b_keff[16], b_gflops[16], b_cos[16];

                snprintf(b_m, sizeof(b_m), "%d", M);
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_int8, sizeof(b_int8), "%.1f", r.int8_min_us);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                snprintf(b_gflops, sizeof(b_gflops), "%.0f", r.gflops);
                snprintf(b_cos, sizeof(b_cos), "%.6f", r.cosine_sim);

                const char *gate = "";
                if (r.speedup_vs_int8 >= SPEEDUP_GATE && r.correctness_pass)
                    gate = "PASS ✓";
                else if (!r.correctness_pass)
                    gate = "COS ✗";
                else
                    gate = "PERF ✗";

                table << r.format_name << b_m
                      << b_min << b_int8 << b_speedup
                      << b_theo << b_keff
                      << b_gflops << b_cos << gate << fort::endr;

                EXPECT_GE(r.cosine_sim, COSINE_SIM_GATE)
                    << fmt.name << " M=" << M
                    << " cosine=" << r.cosine_sim;
            }
            table << fort::separator;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

} // anonymous namespace
