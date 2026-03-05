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
 *   - Cosine similarity: GPU vs HipBLAS FP32 reference (correctness gate)
 *
 * Multi-GPU support: work items are distributed across all available GPUs
 * using cost-descending round-robin to balance load evenly.
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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
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

    // =============================================================================
    // Constants
    // =============================================================================

    // MI50/MI60 theoretical HBM2 bandwidth (GB/s)
    // MI60 = 1.0 TB/s, MI50 = 0.77 TB/s. Use MI60 as reference.
    constexpr double HBM2_PEAK_GBPS = 1000.0;

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    /// Correctness gate: cosine similarity between native-VNNI and FP32 reference.
    /// IQ1_M uses a relaxed threshold because its {-1,0,+1} quantization is
    /// inherently lossy — observed cosine is 0.983-0.987, which is correct
    /// behavior for a 1.9 BPW format.
    constexpr float COSINE_SIM_GATE = 0.99f;
    constexpr float COSINE_SIM_GATE_IQ1_M = 0.98f;

    inline float cosine_gate_for(const std::string &format_name)
    {
        return (format_name == "IQ1_M") ? COSINE_SIM_GATE_IQ1_M : COSINE_SIM_GATE;
    }

    /// Number of GPUs to use (auto-detected, capped at available)
    static int NUM_GPUS = 1;

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

        // Correctness (GPU-based HipBLAS reference)
        float cosine_sim = 0.0f;
        bool correctness_pass = false;
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
                NUM_GPUS = std::min(device_count, 3);
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
        /// Time a GEMV kernel on a specific device. Returns sorted timing vector in μs.
        static std::vector<double> timeKernel(ROCmQuantisedGemmKernel &kernel,
                                              TensorBase *input, TensorBase *output,
                                              int N, int K, int device_id)
        {
            (void)hipSetDevice(device_id);
            const int M = 1;

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

        /// Benchmark INT8 VNNI reference (Q8_0 → INT8 scatter GEMV) for a shape.
        /// Thread-safe: creates all resources locally.
        /// Returns min kernel time in μs.
        static double benchmarkINT8Reference(const GEMVShape &shape, int device_id)
        {
            (void)hipSetDevice(device_id);
            const int M = 1;

            // Create Q8_0 weights — packs to INT8 VNNI (no native-VNNI payload)
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return 0.0;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return 0.0;

            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
                return 0.0;
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
                return 0.0;
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
                return 0.0;

            auto times = timeKernel(kernel, input.get(), output.get(),
                                    shape.N, shape.K, device_id);
            kernel.unbindWorkspace();
            return times.empty() ? 0.0 : times.front();
        }

        /// Run a single format+shape benchmark on a specific device.
        /// Thread-safe: does not use gtest assertions internally.
        static BenchResult benchmarkFormat(const PerfFormatSpec &fmt,
                                           const GEMVShape &shape,
                                           double int8_ref_us,
                                           TensorBase *weights,
                                           const GpuWeightsCache *gpu_weights,
                                           int device_id)
        {
            (void)hipSetDevice(device_id);

            BenchResult result{};
            result.format_name = fmt.name;
            result.bpw = fmt.bpw;
            result.shape_name = shape.name;
            result.N = shape.N;
            result.K = shape.K;

            const int M = 1;

            if (!weights)
                return result;

            // 1. Pack pre-created quantized weights
            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights, packed))
                return result;
            if (packed.native_vnni_payload.empty())
                return result;

            // Calculate weight bytes (native-VNNI payload + scales + mins)
            result.weight_bytes =
                static_cast<double>(packed.native_vnni_payload.size()) +
                static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t)) +
                static_cast<double>(packed.native_vnni_mins.size() * sizeof(uint16_t));

            // 2. Create kernel + workspace
            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
                return result;
            kernel.bindWorkspace(workspace.get());

            // 3. Create input/output tensors and upload to GPU
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                return result;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                return result;
            }

            // 4. Correctness: GPU-based FP32 reference via hipBLAS
            {
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                if (gpu_weights && gpu_weights->d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input.get());
                    const float *d_input = reinterpret_cast<const float *>(
                        in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        const size_t out_elems = static_cast<size_t>(shape.N);
                        float *d_ref_output = nullptr;
                        auto hip_err = hipMalloc(&d_ref_output, out_elems * sizeof(float));
                        if (hip_err == hipSuccess)
                        {
                            // hipBLAS GEMM with M=1 is effectively GEMV
                            bool gemm_ok = gpuReferenceFP32Gemm(
                                d_input, gpu_weights->d_weights,
                                d_ref_output, M, shape.N, shape.K, device_id);
                            (void)hipDeviceSynchronize();

                            if (gemm_ok)
                            {
                                const float *d_gpu_output = reinterpret_cast<const float *>(
                                    dynamic_cast<FP32Tensor *>(output.get())
                                        ->gpu_data_ptr());
                                result.cosine_sim = gpuCosineSimilarity(
                                    d_gpu_output, d_ref_output, out_elems, device_id);
                                result.correctness_pass =
                                    (result.cosine_sim >= cosine_gate_for(result.format_name));
                            }
                            (void)hipFree(d_ref_output);
                        }
                    }
                }

                // Re-upload output for timed runs
                output->ensureOnDevice(DeviceId::rocm(device_id));
            }

            // 5. Timed runs
            auto times = timeKernel(kernel, input.get(), output.get(),
                                    shape.N, shape.K, device_id);

            double max_us;
            computeStats(times, result.mean_us, result.min_us, max_us, result.stddev_us);

            // 6. Compute metrics
            result.eff_bw_gbps = (result.weight_bytes / (result.min_us * 1e-6)) / 1e9;
            result.bw_efficiency = (result.eff_bw_gbps / HBM2_PEAK_GBPS) * 100.0;
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
    // Test: Single-shape sweep across all 18 formats (quick CI check)
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

        // INT8 reference
        double int8_us = benchmarkINT8Reference(shape, 0);
        fprintf(stderr, "[NativeVNNI Perf] INT8 reference: %.1f μs\n", int8_us);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight KB" << "Min μs" << "Mean μs"
              << "Speedup" << "Kern Eff" << "BW GB/s" << "BW Eff %" << "Cosine"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            auto weights = fmt.create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            GpuWeightsCache gpu_w;
            if (weights)
            {
                std::vector<float> w_fp32(static_cast<size_t>(shape.N) * shape.K);
                weights->to_fp32(w_fp32.data());
                gpu_w.upload(w_fp32.data(), shape.N, shape.K, 0);
            }

            auto r = benchmarkFormat(fmt, shape, int8_us, weights.get(), &gpu_w, 0);

            char buf_bpw[16], buf_kb[16], buf_min[16], buf_mean[16];
            char buf_speedup[16], buf_keff[16], buf_bw[16], buf_eff[16], buf_cos[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_kb, sizeof(buf_kb), "%.0f", r.weight_bytes / 1024.0);
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_mean, sizeof(buf_mean), "%.1f", r.mean_us);
            snprintf(buf_speedup, sizeof(buf_speedup), "%.2fx", r.speedup_vs_int8);
            snprintf(buf_keff, sizeof(buf_keff), "%.0f%%", r.kernel_efficiency);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_cos, sizeof(buf_cos), "%.4f", r.cosine_sim);
            table << r.format_name << buf_bpw << buf_kb << buf_min << buf_mean
                  << buf_speedup << buf_keff << buf_bw << buf_eff << buf_cos
                  << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    // =============================================================================
    // Test: Full matrix — all formats × all shapes — multi-GPU
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
        fprintf(stderr, "[NativeVNNI Perf] Using %d GPU(s) for parallel benchmarking\n", NUM_GPUS);

        // =========================================================================
        // Phase 1: Benchmark INT8 VNNI reference for each shape — multi-GPU
        // =========================================================================
        fprintf(stderr, "\n[Phase 1] Benchmarking INT8 VNNI reference on %d GPU(s)...\n",
                NUM_GPUS);

        struct Int8Work
        {
            int shape_idx;
            double cost;
        };
        std::vector<Int8Work> int8_work;
        for (int si = 0; si < (int)SHAPES.size(); ++si)
            int8_work.push_back({si, (double)SHAPES[si].N * SHAPES[si].K});

        std::sort(int8_work.begin(), int8_work.end(),
                  [](const Int8Work &a, const Int8Work &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<Int8Work>> int8_per_gpu(NUM_GPUS);
        for (size_t i = 0; i < int8_work.size(); ++i)
            int8_per_gpu[i % NUM_GPUS].push_back(int8_work[i]);

        std::vector<double> int8_times(SHAPES.size(), 0.0);
        std::atomic<int> int8_done{0};

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    for (const auto &w : int8_per_gpu[g])
                    {
                        double ref_us = benchmarkINT8Reference(SHAPES[w.shape_idx], g);
                        int8_times[w.shape_idx] = ref_us;
                        int done = ++int8_done;
                        fprintf(stderr, "  [GPU %d] INT8 %s: %.1f μs  (%d/%zu)\n",
                                g, SHAPES[w.shape_idx].name.c_str(), ref_us,
                                done, int8_work.size());
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        std::unordered_map<std::string, double> int8_ref_us;
        for (int si = 0; si < (int)SHAPES.size(); ++si)
            int8_ref_us[SHAPES[si].name] = int8_times[si];

        // =========================================================================
        // Phase 2: Benchmark all native-VNNI formats — multi-GPU
        // =========================================================================
        fprintf(stderr, "\n[Phase 2] Benchmarking %zu native-VNNI formats on %d GPU(s)...\n",
                ALL_PERF_FORMATS.size(), NUM_GPUS);

        struct WorkGroup
        {
            int format_idx;
            int shape_idx;
            double cost;
        };
        std::vector<WorkGroup> groups;
        for (int fi = 0; fi < (int)ALL_PERF_FORMATS.size(); ++fi)
            for (int si = 0; si < (int)SHAPES.size(); ++si)
                groups.push_back({fi, si, (double)SHAPES[si].N * SHAPES[si].K});

        std::sort(groups.begin(), groups.end(),
                  [](const WorkGroup &a, const WorkGroup &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<WorkGroup>> per_gpu(NUM_GPUS);
        for (size_t i = 0; i < groups.size(); ++i)
            per_gpu[i % NUM_GPUS].push_back(groups[i]);

        const size_t num_shapes = SHAPES.size();
        const size_t total = ALL_PERF_FORMATS.size() * num_shapes;
        std::vector<BenchResult> results(total);
        std::atomic<int> phase2_done{0};
        const size_t total_groups = groups.size();

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    (void)hipSetDevice(g);
                    for (const auto &wg : per_gpu[g])
                    {
                        const auto &fmt = ALL_PERF_FORMATS[wg.format_idx];
                        const auto &shape = SHAPES[wg.shape_idx];

                        auto weights = fmt.create(
                            static_cast<size_t>(shape.N),
                            static_cast<size_t>(shape.K));
                        GpuWeightsCache gpu_w;
                        if (weights)
                        {
                            std::vector<float> w_fp32(
                                static_cast<size_t>(shape.N) * shape.K);
                            weights->to_fp32(w_fp32.data());
                            gpu_w.upload(w_fp32.data(), shape.N, shape.K, g);
                        }

                        double ref_us = int8_ref_us.count(shape.name)
                                            ? int8_ref_us[shape.name]
                                            : 0.0;

                        auto r = benchmarkFormat(fmt, shape, ref_us,
                                                 weights.get(), &gpu_w, g);

                        size_t idx = wg.format_idx * num_shapes + wg.shape_idx;
                        results[idx] = std::move(r);

                        int done = ++phase2_done;
                        fprintf(stderr, "  [GPU %d] %s/%s %.1f μs cos=%.4f %s  (%d/%zu)\n",
                                g, fmt.name.c_str(), shape.name.c_str(),
                                results[idx].min_us, results[idx].cosine_sim,
                                results[idx].correctness_pass ? "✓" : "✗",
                                done, total_groups);
                    } });
            }
            for (auto &t : threads)
                t.join();
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
                  << "BW GB/s" << "BW Eff %" << "Cosine" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 9; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            for (const auto &r : results)
            {
                if (r.shape_name != shape.name)
                    continue;

                char b_bpw[16], b_kb[16], b_min[16];
                char b_speedup[16], b_theo[16], b_keff[16];
                char b_bw[16], b_bweff[16], b_cos[16];

                snprintf(b_bpw, sizeof(b_bpw), "%.1f", r.bpw);
                snprintf(b_kb, sizeof(b_kb), "%.0f", r.weight_bytes / 1024.0);
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                snprintf(b_bw, sizeof(b_bw), "%.1f", r.eff_bw_gbps);
                snprintf(b_bweff, sizeof(b_bweff), "%.1f%%", r.bw_efficiency);
                snprintf(b_cos, sizeof(b_cos), "%.4f", r.cosine_sim);

                table << r.format_name << b_bpw << b_kb << b_min
                      << b_speedup << b_theo << b_keff
                      << b_bw << b_bweff << b_cos << fort::endr;
            }

            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }

        // =========================================================================
        // Phase 4: Grand Summary — average across all shapes, sorted by kern eff
        // =========================================================================
        fprintf(stderr, "\n");
        fort::utf8_table summary;
        summary.set_border_style(FT_DOUBLE2_STYLE);
        summary << fort::header
                << "Format" << "BPW" << "Avg Min μs" << "Avg Speedup"
                << "Theoretical" << "Avg Kern Eff" << "Avg BW GB/s"
                << "Avg Cosine" << "Status"
                << fort::endr;

        summary.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            summary.column(c).set_cell_text_align(fort::text_align::right);

        struct FormatSummary
        {
            std::string name;
            double bpw;
            double avg_min_us;
            double avg_speedup;
            double theoretical;
            double avg_kern_eff;
            double avg_bw;
            double avg_cosine;
            bool all_pass;
        };
        std::vector<FormatSummary> format_summaries;

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            double total_min = 0.0, total_speedup = 0.0, total_keff = 0.0;
            double total_bw = 0.0, total_cos = 0.0;
            bool all_pass = true;
            int count = 0;
            for (const auto &r : results)
            {
                if (r.format_name == fmt.name)
                {
                    total_min += r.min_us;
                    total_speedup += r.speedup_vs_int8;
                    total_keff += r.kernel_efficiency;
                    total_bw += r.eff_bw_gbps;
                    total_cos += r.cosine_sim;
                    if (!r.correctness_pass)
                        all_pass = false;
                    ++count;
                }
            }
            if (count == 0)
                continue;

            format_summaries.push_back({
                fmt.name,
                fmt.bpw,
                total_min / count,
                total_speedup / count,
                8.0 / fmt.bpw,
                total_keff / count,
                total_bw / count,
                total_cos / count,
                all_pass,
            });
        }

        // Sort by kernel efficiency ascending (worst first) for tuning focus
        std::sort(format_summaries.begin(), format_summaries.end(),
                  [](const FormatSummary &a, const FormatSummary &b)
                  { return a.avg_kern_eff < b.avg_kern_eff; });

        for (const auto &fs : format_summaries)
        {
            char b_bpw[16], b_min[16], b_speedup[16], b_theo[16];
            char b_keff[16], b_bw[16], b_cos[16];
            snprintf(b_bpw, sizeof(b_bpw), "%.1f", fs.bpw);
            snprintf(b_min, sizeof(b_min), "%.1f", fs.avg_min_us);
            snprintf(b_speedup, sizeof(b_speedup), "%.2fx", fs.avg_speedup);
            snprintf(b_theo, sizeof(b_theo), "%.2fx", fs.theoretical);
            snprintf(b_keff, sizeof(b_keff), "%.0f%%", fs.avg_kern_eff);
            snprintf(b_bw, sizeof(b_bw), "%.1f", fs.avg_bw);
            snprintf(b_cos, sizeof(b_cos), "%.4f", fs.avg_cosine);
            const char *status = fs.all_pass ? "✓" : "✗";

            summary << fs.name << b_bpw << b_min << b_speedup
                    << b_theo << b_keff << b_bw << b_cos << status
                    << fort::endr;
        }

        fprintf(stderr, "GRAND SUMMARY: Average across all shapes (sorted by Kern Eff ascending — worst first)\n");
        fprintf(stderr, "%s\n", summary.to_string().c_str());
        fprintf(stderr, "Speedup = INT8_time / format_time (>1x = faster than INT8)\n");
        fprintf(stderr, "Theoretical = 8.0/BPW (ideal speedup from bandwidth savings alone)\n");
        fprintf(stderr, "Kern Eff = Speedup/Theoretical × 100%% (how close to bandwidth-optimal)\n");
        fprintf(stderr, "Cosine = GPU output vs HipBLAS FP32 reference (gate: >= %.2f, IQ1_M: >= %.2f)\n",
                COSINE_SIM_GATE, COSINE_SIM_GATE_IQ1_M);

        // Validate correctness (per-format gate)
        for (const auto &r : results)
        {
            const float gate = cosine_gate_for(r.format_name);
            EXPECT_GE(r.cosine_sim, gate)
                << r.format_name << "/" << r.shape_name
                << " cosine=" << r.cosine_sim
                << " (gate=" << gate << ")";
        }
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
              << "BW Eff %" << "Bytes/Elem" << "Cosine" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 7; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &sel_name : selected)
        {
            auto it = std::find_if(ALL_PERF_FORMATS.begin(), ALL_PERF_FORMATS.end(),
                                   [&](const PerfFormatSpec &f)
                                   { return f.name == sel_name; });
            if (it == ALL_PERF_FORMATS.end())
                continue;

            auto weights = it->create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            GpuWeightsCache gpu_w;
            if (weights)
            {
                std::vector<float> w_fp32(static_cast<size_t>(shape.N) * shape.K);
                weights->to_fp32(w_fp32.data());
                gpu_w.upload(w_fp32.data(), shape.N, shape.K, 0);
            }

            auto r = benchmarkFormat(*it, shape, 0.0, weights.get(), &gpu_w, 0);

            double bytes_per_elem = r.weight_bytes / (static_cast<double>(r.N) * r.K);

            char buf_bpw[16], buf_mb[16], buf_min[16], buf_bw[16];
            char buf_eff[16], buf_bpe[16], buf_cos[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_mb, sizeof(buf_mb), "%.2f", r.weight_bytes / (1024.0 * 1024.0));
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_bpe, sizeof(buf_bpe), "%.3f", bytes_per_elem);
            snprintf(buf_cos, sizeof(buf_cos), "%.4f", r.cosine_sim);

            table << r.format_name << buf_bpw << buf_mb << buf_min << buf_bw
                  << buf_eff << buf_bpe << buf_cos << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
        fprintf(stderr, "Expected: lower BPW = less data = lower μs (if decode ALU < BW savings)\n");
        fprintf(stderr, "HBM2 peak bandwidth reference: %.0f GB/s\n", HBM2_PEAK_GBPS);
#endif
    }

} // anonymous namespace
