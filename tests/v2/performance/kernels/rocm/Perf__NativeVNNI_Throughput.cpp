/**
 * @file Perf__NativeVNNI_Throughput.cpp
 * @brief Per-format bandwidth benchmark for native-VNNI GEMV kernels
 *
 * Measures decode throughput (M=1 GEMV) for all 18 native-VNNI formats
 * at model-realistic dimensions (Qwen2.5-0.5B, 3B, and 7B layer shapes).
 *
 * Each sub-8-bit format is benchmarked against the INT8 VNNI reference
 * (Q8_0 packed to INT8 scatter GEMV) on the same shape, yielding:
 *
 * Metrics reported:
 *   - Kernel time (μs): min/mean across benchmark runs
 *   - Effective bandwidth (GB/s): weight_bytes_read / kernel_time
 *   - BW efficiency (%): effective_BW / HBM_peak_BW
 *   - Speedup vs INT8: int8_min_us / format_min_us
 *   - Theoretical speedup: 8.0 / bpw (from streaming fewer bytes)
 *   - Kernel efficiency: actual_speedup / theoretical_speedup × 100%
 *
 * The benchmark uses multiply_tensor() which includes:
 *   1. FP32→INT8 activation quantization on GPU
 *   2. Native-VNNI kernel dispatch (or INT8 scatter GEMV for reference)
 *   3. Scale application (FP32 output)
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

    // MI50/MI60 theoretical HBM2 bandwidth (GB/s)
    // MI60 = 1.0 TB/s, MI50 = 0.77 TB/s. Use MI60 as reference.
    constexpr double HBM2_PEAK_GBPS = 1000.0;

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    // =============================================================================
    // Format descriptors
    // =============================================================================

    struct PerfFormatSpec
    {
        std::string name;
        double bpw;         ///< Bits per weight element
        bool is_superblock; ///< K must be multiple of 256

        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    static const std::vector<PerfFormatSpec> ALL_PERF_FORMATS = {
        // Tier 1: Simple 32-element blocks
        {"Q4_0", 4.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", 4.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"Q4_1", 5.0, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q5_0", 5.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", 6.0, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},

        // Tier 1 super-block
        {"IQ4_XS", 4.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_XSRandom({N, K}); }},

        // Tier 2: K-quant super-blocks
        {"Q4_K", 4.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_K", 5.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", 6.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", 3.4, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", 2.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},

        // Tier 3: IQ grid-index super-blocks
        {"IQ3_S", 3.4, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", 3.1, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", 2.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", 2.3, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", 2.1, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},

        // Tier 4: IQ1 ultra-low-bit grid-index super-blocks
        {"IQ1_S", 1.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", 1.9, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
    };

    // Model-realistic GEMV shapes (N×K)
    struct GEMVShape
    {
        std::string name;
        int N;
        int K;
    };

    // Qwen2.5-0.5B:  hidden=896,  intermediate=4864
    // Qwen2.5-3B:    hidden=2048, intermediate=11008
    // Qwen2.5-7B:    hidden=3584, intermediate=18944
    // All K values are multiples of 32 (minimum block size).
    // Super-block formats (256-element) handle non-256-aligned K via sub-block iteration.
    static const std::vector<GEMVShape> SHAPES = {
        // Qwen2.5-0.5B
        {"0.5B_AttnOut", 896, 896},    // Qwen2.5-0.5B attention output projection
        {"0.5B_QKV", 896 * 3, 896},    // Qwen2.5-0.5B attention QKV projection
        {"0.5B_FFN_Up", 4864, 896},    // Qwen2.5-0.5B FFN gate/up
        {"0.5B_FFN_Dn", 896, 4864},    // Qwen2.5-0.5B FFN down
        {"0.5B_LM_Head", 151936, 896}, // Qwen2.5-0.5B LM head (vocab projection)
        // Qwen2.5-3B
        {"3B_AttnOut", 2048, 2048},   // Qwen2.5-3B attention output projection
        {"3B_FFN_Up", 11008, 2048},   // Qwen2.5-3B FFN gate/up
        {"3B_FFN_Dn", 2048, 11008},   // Qwen2.5-3B FFN down
        {"3B_LM_Head", 151936, 2048}, // Qwen2.5-3B LM head (vocab projection)
        // Qwen2.5-7B
        {"7B_QKV", 3584 * 3, 3584}, // Qwen2.5-7B attention projection
        {"7B_FFN_Up", 18944, 3584}, // Qwen2.5-7B FFN gate/up
        {"7B_FFN_Dn", 3584, 18944}, // Qwen2.5-7B FFN down
    };

    // =============================================================================
    // Benchmark result
    // =============================================================================

    struct BenchResult
    {
        std::string format_name;
        double bpw;
        std::string shape_name;
        int N, K;

        double min_us;
        double mean_us;
        double stddev_us;

        double weight_bytes;  // native-VNNI payload + scales + mins bytes
        double eff_bw_gbps;   // effective bandwidth at min time
        double bw_efficiency; // % of HBM peak

        // INT8 reference comparison (populated when reference available)
        double int8_min_us = 0.0;         // INT8 VNNI reference min time for same shape
        double speedup_vs_int8 = 0.0;     // int8_min_us / min_us (>1 = faster than INT8)
        double theoretical_speedup = 0.0; // 8.0 / bpw (expected from bandwidth savings)
        double kernel_efficiency = 0.0;   // (speedup_vs_int8 / theoretical_speedup) * 100%
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
    // Test fixture
    // =============================================================================

    class NativeVNNIPerfTest : public ::testing::Test
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
        /// Time a kernel+workspace pair for a given shape. Returns min time in μs.
        /// Shared between native-VNNI and INT8 reference benchmarking.
        double timeKernel(ROCmQuantisedGemmKernel &kernel, const GEMVShape &shape,
                          TensorBase *input, TensorBase *output)
        {
            const int M = 1;

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input, output, M, shape.N, shape.K);
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
                kernel.multiply_tensor(input, output, M, shape.N, shape.K);
                (void)hipEventRecord(stop);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);

            double mean, min_val, max_val, stddev;
            computeStats(times_us, mean, min_val, max_val, stddev);
            return min_val;
        }

        /// Benchmark INT8 VNNI reference (Q8_0 → INT8 scatter GEMV) for a shape.
        /// Returns min kernel time in μs.
        double benchmarkINT8Reference(const GEMVShape &shape)
        {
            const int M = 1;

            // Create Q8_0 weights — packs to INT8 VNNI (no native-VNNI payload)
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return 0.0;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return 0.0;

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            if (!workspace->allocate(reqs))
                return 0.0;
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(0)))
                return 0.0;
            if (!output->allocateOnDevice(DeviceId::rocm(0)))
                return 0.0;

            double min_us = timeKernel(kernel, shape, input.get(), output.get());
            kernel.unbindWorkspace();
            return min_us;
        }

        /// Run a single format+shape benchmark, returning the result.
        BenchResult benchmarkFormat(const PerfFormatSpec &fmt,
                                    const GEMVShape &shape)
        {
            BenchResult result{};
            result.format_name = fmt.name;
            result.bpw = fmt.bpw;
            result.shape_name = shape.name;
            result.N = shape.N;
            result.K = shape.K;

            const int M = 1;

            // 1. Create quantized weights and pack
            auto weights = fmt.create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            EXPECT_NE(weights, nullptr);
            if (!weights)
                return result;

            ROCmPackedWeights packed;
            EXPECT_TRUE(packWeightsToROCm(weights.get(), packed));
            if (packed.native_vnni_payload.empty())
                return result;

            // Calculate weight bytes (native-VNNI payload + scales + mins)
            result.weight_bytes = static_cast<double>(packed.native_vnni_payload.size()) + static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t)) + static_cast<double>(packed.native_vnni_mins.size() * sizeof(uint16_t));

            // 2. Create kernel + workspace (budget computed from actual requirements)
            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            EXPECT_TRUE(workspace->allocate(reqs));
            kernel.bindWorkspace(workspace.get());

            // 3. Create input/output tensors and upload to GPU
            //    (essential: multiply_tensor uses gpu_data_ptr() to detect GPU-resident
            //     tensors and route to the M==1 GEMV fast path with native-VNNI)
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            EXPECT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
            EXPECT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

            // 4+5. Timed runs
            result.min_us = timeKernel(kernel, shape, input.get(), output.get());

            // Compute full stats from a second timed pass for mean/stddev
            {
                std::vector<double> times_us;
                times_us.reserve(BENCH_RUNS);
                hipEvent_t ev_start = nullptr, ev_stop = nullptr;
                (void)hipEventCreate(&ev_start);
                (void)hipEventCreate(&ev_stop);

                for (int i = 0; i < BENCH_RUNS; ++i)
                {
                    (void)hipDeviceSynchronize();
                    (void)hipEventRecord(ev_start);
                    kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                    (void)hipEventRecord(ev_stop);
                    (void)hipEventSynchronize(ev_stop);

                    float ms = 0.0f;
                    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
                    times_us.push_back(static_cast<double>(ms) * 1000.0);
                }

                (void)hipEventDestroy(ev_start);
                (void)hipEventDestroy(ev_stop);

                double max_us;
                computeStats(times_us, result.mean_us, result.min_us, max_us, result.stddev_us);
            }

            // 6. Compute stats
            result.eff_bw_gbps = (result.weight_bytes / (result.min_us * 1e-6)) / 1e9;
            result.bw_efficiency = (result.eff_bw_gbps / HBM2_PEAK_GBPS) * 100.0;
            result.theoretical_speedup = 8.0 / result.bpw;

            kernel.unbindWorkspace();
            return result;
        }
#endif
    };

    // =============================================================================
    // Test: Single-shape sweep across all 16 formats (quick CI check)
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, AllFormats_0_5B_FFN_Up)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const GEMVShape shape{"0.5B_FFN_Up", 4864, 896};

        fprintf(stderr, "\n[NativeVNNI Perf] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI Perf] Shape: %s (N=%d K=%d) | %d warmup + %d runs\n",
                shape.name.c_str(), shape.N, shape.K, WARMUP_RUNS, BENCH_RUNS);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight KB" << "Min μs" << "Mean μs"
              << "BW GB/s" << "BW Eff %" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            auto r = benchmarkFormat(fmt, shape);
            char buf_bpw[16], buf_kb[16], buf_min[16], buf_mean[16], buf_bw[16], buf_eff[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_kb, sizeof(buf_kb), "%.0f", r.weight_bytes / 1024.0);
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_mean, sizeof(buf_mean), "%.1f", r.mean_us);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            table << r.format_name << buf_bpw << buf_kb << buf_min << buf_mean
                  << buf_bw << buf_eff << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    // =============================================================================
    // Test: Full matrix — all formats × all shapes
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, AllFormats_AllShapes_Matrix)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI Perf] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI Perf] %zu formats × %zu shapes | %d warmup + %d runs each\n",
                ALL_PERF_FORMATS.size(), SHAPES.size(), WARMUP_RUNS, BENCH_RUNS);

        // =========================================================================
        // Phase 1: Benchmark INT8 VNNI reference for each shape
        // =========================================================================
        fprintf(stderr, "\n[NativeVNNI Perf] Benchmarking INT8 VNNI reference...\n");
        std::unordered_map<std::string, double> int8_ref_us;
        for (const auto &shape : SHAPES)
        {
            double ref_us = benchmarkINT8Reference(shape);
            int8_ref_us[shape.name] = ref_us;
            fprintf(stderr, "  INT8 ref %s: %.1f μs\n", shape.name.c_str(), ref_us);
        }

        // =========================================================================
        // Phase 2: Benchmark all native-VNNI formats
        // =========================================================================
        fprintf(stderr, "\n[NativeVNNI Perf] Benchmarking %zu native-VNNI formats...\n",
                ALL_PERF_FORMATS.size());

        std::vector<BenchResult> results;
        results.reserve(ALL_PERF_FORMATS.size() * SHAPES.size());

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            for (const auto &shape : SHAPES)
            {
                auto r = benchmarkFormat(fmt, shape);

                // Populate INT8 comparison fields
                auto it = int8_ref_us.find(shape.name);
                if (it != int8_ref_us.end() && it->second > 0.0 && r.min_us > 0.0)
                {
                    r.int8_min_us = it->second;
                    r.speedup_vs_int8 = it->second / r.min_us;
                    r.theoretical_speedup = 8.0 / r.bpw;
                    r.kernel_efficiency = (r.speedup_vs_int8 / r.theoretical_speedup) * 100.0;
                }
                results.push_back(std::move(r));
            }
        }

        // =========================================================================
        // Phase 3: Print per-shape comparison tables
        // =========================================================================
        for (const auto &shape : SHAPES)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            auto ref_it = int8_ref_us.find(shape.name);
            double ref_us = (ref_it != int8_ref_us.end()) ? ref_it->second : 0.0;

            char title[256];
            snprintf(title, sizeof(title),
                     "Shape: %s (N=%d K=%d) | INT8 ref: %.1f μs",
                     shape.name.c_str(), shape.N, shape.K, ref_us);

            table << fort::header
                  << "Format" << "BPW" << "Wt KB" << "Min μs"
                  << "Speedup" << "Theoret." << "Kern Eff"
                  << "BW GB/s" << "BW Eff %" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 8; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            for (const auto &r : results)
            {
                if (r.shape_name != shape.name)
                    continue;

                char b_bpw[16], b_kb[16], b_min[16];
                char b_speedup[16], b_theo[16], b_keff[16];
                char b_bw[16], b_bweff[16];

                snprintf(b_bpw, sizeof(b_bpw), "%.1f", r.bpw);
                snprintf(b_kb, sizeof(b_kb), "%.0f", r.weight_bytes / 1024.0);
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                snprintf(b_bw, sizeof(b_bw), "%.1f", r.eff_bw_gbps);
                snprintf(b_bweff, sizeof(b_bweff), "%.1f%%", r.bw_efficiency);

                table << r.format_name << b_bpw << b_kb << b_min
                      << b_speedup << b_theo << b_keff
                      << b_bw << b_bweff << fort::endr;
            }

            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }

        // =========================================================================
        // Phase 4: Summary — average across all shapes
        // =========================================================================
        fprintf(stderr, "\n");
        fort::utf8_table summary;
        summary.set_border_style(FT_DOUBLE2_STYLE);
        summary << fort::header
                << "Format" << "BPW" << "Avg Min μs" << "Avg Speedup"
                << "Theoretical" << "Avg Kern Eff" << "Avg BW GB/s"
                << fort::endr;

        summary.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 6; ++c)
            summary.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            double total_min = 0.0, total_speedup = 0.0, total_keff = 0.0, total_bw = 0.0;
            int count = 0;
            for (const auto &r : results)
            {
                if (r.format_name == fmt.name)
                {
                    total_min += r.min_us;
                    total_speedup += r.speedup_vs_int8;
                    total_keff += r.kernel_efficiency;
                    total_bw += r.eff_bw_gbps;
                    ++count;
                }
            }
            if (count == 0)
                continue;

            char b_bpw[16], b_min[16], b_speedup[16], b_theo[16], b_keff[16], b_bw[16];
            snprintf(b_bpw, sizeof(b_bpw), "%.1f", fmt.bpw);
            snprintf(b_min, sizeof(b_min), "%.1f", total_min / count);
            snprintf(b_speedup, sizeof(b_speedup), "%.2fx", total_speedup / count);
            snprintf(b_theo, sizeof(b_theo), "%.2fx", 8.0 / fmt.bpw);
            snprintf(b_keff, sizeof(b_keff), "%.0f%%", total_keff / count);
            snprintf(b_bw, sizeof(b_bw), "%.1f", total_bw / count);

            summary << fmt.name << b_bpw << b_min << b_speedup
                    << b_theo << b_keff << b_bw << fort::endr;
        }

        fprintf(stderr, "SUMMARY: Average across all shapes (vs INT8 VNNI reference)\n%s\n",
                summary.to_string().c_str());
        fprintf(stderr, "Speedup = INT8_time / format_time (>1x = faster than INT8)\n");
        fprintf(stderr, "Theoretical = 8.0/BPW (ideal speedup from bandwidth savings alone)\n");
        fprintf(stderr, "Kern Eff = Speedup/Theoretical × 100%% (how close to bandwidth-optimal)\n");
#endif
    }

    // =============================================================================
    // Test: BPW-vs-bandwidth scaling curve (focused)
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, BPW_Scaling_7B_FFN_Down)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        // Largest realistic shape — best for exposing bandwidth differences
        const GEMVShape shape{"7B_FFN_Dn", 3584, 18944};

        fprintf(stderr, "\n[NativeVNNI Perf] BPW Scaling Curve\n");
        fprintf(stderr, "[NativeVNNI Perf] Shape: %s (N=%d K=%d) — largest GEMV shape\n",
                shape.name.c_str(), shape.N, shape.K);

        // Select representative formats spanning the BPW range
        const std::vector<std::string> selected = {
            "IQ2_XXS",
            "IQ2_XS",
            "Q2_K",
            "IQ3_XXS",
            "Q3_K",
            "Q4_0",
            "Q4_K",
            "Q5_0",
            "Q5_K",
            "Q6_K",
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight MB" << "Min μs" << "BW GB/s"
              << "BW Eff %" << "Bytes/Elem" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 6; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &sel_name : selected)
        {
            auto it = std::find_if(ALL_PERF_FORMATS.begin(), ALL_PERF_FORMATS.end(),
                                   [&](const PerfFormatSpec &f)
                                   { return f.name == sel_name; });
            if (it == ALL_PERF_FORMATS.end())
                continue;

            auto r = benchmarkFormat(*it, shape);

            double bytes_per_elem = r.weight_bytes / (static_cast<double>(r.N) * r.K);

            char buf_bpw[16], buf_mb[16], buf_min[16], buf_bw[16], buf_eff[16], buf_bpe[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_mb, sizeof(buf_mb), "%.2f", r.weight_bytes / (1024.0 * 1024.0));
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_bpe, sizeof(buf_bpe), "%.3f", bytes_per_elem);

            table << r.format_name << buf_bpw << buf_mb << buf_min << buf_bw
                  << buf_eff << buf_bpe << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
        fprintf(stderr, "Expected: lower BPW = less data = lower μs (if decode ALU < BW savings)\n");
        fprintf(stderr, "HBM2 peak bandwidth reference: %.0f GB/s\n", HBM2_PEAK_GBPS);
#endif
    }

} // anonymous namespace
