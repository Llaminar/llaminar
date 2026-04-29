/**
 * @file Perf__Q8_0_NativeVNNI_GemvTuning.cpp
 * @brief Q8_0 decode GEMV tuning sweep: native-VNNI vs INT8-VNNI comparison.
 *
 * Investigates the Q8_0 decode regression when routed through the native-VNNI
 * kernel path (codebook_id=19) instead of the INT8-VNNI scatter GEMV path.
 *
 * Sweeps:
 *   1. INT8-VNNI Q8_0 baseline (packWeightsToROCm → INT8 path)
 *   2. Native-VNNI Q8_0 with default heuristic (TARGET_WAVES_PER_CU=24)
 *   3. Native-VNNI Q8_0 with swept TARGET_WAVES_PER_CU: {4, 8, 12, 16, 20, 24, 32, 40}
 *   4. Native-VNNI Q8_0 with swept KB: {1, 2, 4, 8, 14, 16, 28, 32, 56}
 *
 * Reports:
 *   - Kernel time (μs), effective bandwidth (GB/s), BW efficiency (%)
 *   - Cosine similarity vs HipBLAS FP32 reference (correctness gate)
 *   - Speedup vs INT8-VNNI baseline
 *   - Packing overhead comparison (native VNNI vs INT8 VNNI byte counts)
 *
 * Shapes tested: Qwen2.5-7B layer shapes (the regression model).
 *
 * @note Requires ROCm device. Run with build_v2_release for representative timing.
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
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "GpuVerification.h"
#endif

extern "C"
{
    void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu);
    void rocmGemv_native_vnni_reset_tuning_overrides();
}

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
    constexpr int BENCH_RUNS = 30;
    constexpr float COSINE_SIM_GATE = 0.9990f;

    constexpr double HBM2_PEAK_GBPS = 1000.0; // MI60 theoretical peak

    // =========================================================================
    // Qwen2.5-7B decode shapes (the regression model)
    // =========================================================================

    struct GEMVShape
    {
        std::string name;
        int N;
        int K;
    };

    static const std::vector<GEMVShape> SHAPES_7B = {
        {"7B_QKV", 3584 * 3, 3584},     // Fused QKV projection
        {"7B_AttnOut", 3584, 3584},      // Attention output (Wo)
        {"7B_FFN_GateUp", 18944, 3584},  // Fused gate+up projection
        {"7B_FFN_Down", 3584, 18944},    // Down projection
        {"7B_LM_Head", 152064, 3584},    // LM head (vocab)
    };

    // =========================================================================
    // Result struct
    // =========================================================================

    struct TuningResult
    {
        std::string shape_name;
        std::string variant;
        int N = 0, K = 0;

        // Tuning params
        int kb_forced = -1;
        int target_waves = -1;

        // Timing
        double min_us = 0.0;
        double mean_us = 0.0;
        double stddev_us = 0.0;

        // Data
        double weight_bytes = 0.0;
        double eff_bw_gbps = 0.0;
        double bw_efficiency = 0.0;

        // Correctness
        float cosine_sim = 0.0f;
        bool correctness_pass = false;

        // Comparison
        double speedup_vs_int8 = 0.0;
    };

    // =========================================================================
    // Stats
    // =========================================================================

    static void computeStats(const std::vector<double> &times,
                             double &mean, double &min_val, double &stddev)
    {
        mean = std::accumulate(times.begin(), times.end(), 0.0) /
               static_cast<double>(times.size());
        min_val = *std::min_element(times.begin(), times.end());
        double sq_sum = 0.0;
        for (double t : times)
            sq_sum += (t - mean) * (t - mean);
        stddev = std::sqrt(sq_sum / static_cast<double>(times.size()));
    }

    // =========================================================================
    // Test fixture
    // =========================================================================

    class Q8_0_GEMV_Tuning : public ::testing::Test
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
            rocmGemv_native_vnni_reset_tuning_overrides();
            destroyAllHipBLAS();
#endif
        }

        bool has_device_ = false;
        std::string device_name_;

#ifdef HAVE_ROCM
        /// Time a GEMV kernel. Returns sorted timing vector in μs.
        static std::vector<double> timeKernel(ROCmQuantisedGemmKernel &kernel,
                                              TensorBase *input, TensorBase *output,
                                              int N, int K)
        {
            const int M = 1;

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input, output, M, N, K);
            (void)hipDeviceSynchronize();

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

        /// Benchmark Q8_0 via INT8-VNNI path (packWeightsToROCm standard path).
        static TuningResult benchmarkINT8Path(const GEMVShape &shape,
                                              const GpuWeightsCache *gpu_w)
        {
            TuningResult r;
            r.shape_name = shape.name;
            r.variant = "INT8_VNNI";
            r.N = shape.N;
            r.K = shape.K;

            const int M = 1;

            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return r;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return r;

            // INT8 path: weight_bytes = int8_data_vnni size + scales
            r.weight_bytes = static_cast<double>(packed.int8_data_vnni.size()) +
                             static_cast<double>(packed.scales.size() * sizeof(float));

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            if (!workspace->allocate(reqs))
                return r;
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(0)))
            {
                kernel.unbindWorkspace();
                return r;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(0)))
            {
                kernel.unbindWorkspace();
                return r;
            }

            // Correctness
            {
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                (void)hipDeviceSynchronize();
                output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

                if (gpu_w && gpu_w->d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input.get());
                    const float *d_input = reinterpret_cast<const float *>(in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        const size_t out_elems = static_cast<size_t>(shape.N);
                        float *d_ref = nullptr;
                        if (hipMalloc(&d_ref, out_elems * sizeof(float)) == hipSuccess)
                        {
                            if (gpuReferenceFP32Gemm(d_input, gpu_w->d_weights, d_ref, M, shape.N, shape.K, 0))
                            {
                                (void)hipDeviceSynchronize();
                                r.cosine_sim = gpuCosineSimilarity(
                                    reinterpret_cast<const float *>(
                                        dynamic_cast<FP32Tensor *>(output.get())->gpu_data_ptr()),
                                    d_ref, out_elems, 0);
                                r.correctness_pass = (r.cosine_sim >= COSINE_SIM_GATE);
                            }
                            (void)hipFree(d_ref);
                        }
                    }
                }
                output->ensureOnDevice(DeviceId::rocm(0));
            }

            // Timed runs
            auto times = timeKernel(kernel, input.get(), output.get(), shape.N, shape.K);
            computeStats(times, r.mean_us, r.min_us, r.stddev_us);

            r.eff_bw_gbps = (r.weight_bytes / (r.min_us * 1e-6)) / 1e9;
            r.bw_efficiency = (r.eff_bw_gbps / HBM2_PEAK_GBPS) * 100.0;

            kernel.unbindWorkspace();
            return r;
        }

        /// Benchmark Q8_0 via native-VNNI path (packs to native VNNI format).
        static TuningResult benchmarkNativeVNNIPath(const GEMVShape &shape,
                                                    const GpuWeightsCache *gpu_w,
                                                    int kb_forced = -1,
                                                    int target_waves = -1)
        {
            TuningResult r;
            r.shape_name = shape.name;
            r.N = shape.N;
            r.K = shape.K;
            r.kb_forced = kb_forced;
            r.target_waves = target_waves;

            if (kb_forced > 0 && target_waves > 0)
                r.variant = "NV_kb" + std::to_string(kb_forced) + "_tw" + std::to_string(target_waves);
            else if (kb_forced > 0)
                r.variant = "NV_kb" + std::to_string(kb_forced);
            else if (target_waves > 0)
                r.variant = "NV_tw" + std::to_string(target_waves);
            else
                r.variant = "NativeVNNI_default";

            const int M = 1;

            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return r;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return r;

            // Check if native VNNI payload was produced
            if (packed.native_vnni_payload.empty())
            {
                // packWeightsToROCm routes Q8_0 to INT8 path by default.
                // We need to force native VNNI packing for Q8_0.
                // Use the VNNI format info from Q8_0Tensor to manually pack.
                const auto *fmt = weights->vnniFormatInfo();
                if (!fmt || fmt->codebook_id != 19)
                {
                    fprintf(stderr, "[Q8_0 NativeVNNI] No vnniFormatInfo for Q8_0\n");
                    return r;
                }

                // Pack via TensorBase::packVnniBlock interface
                const size_t N_sz = static_cast<size_t>(shape.N);
                const size_t K_sz = static_cast<size_t>(shape.K);
                const int blocks_per_row = static_cast<int>(K_sz / 32);
                const size_t total_blocks = N_sz * blocks_per_row;
                const int payload_bytes = fmt->payload_bytes; // 32 for Q8_0

                packed.native_vnni_payload.resize(total_blocks * payload_bytes);
                packed.native_vnni_scales.resize(total_blocks);
                packed.native_vnni_codebook_id = fmt->codebook_id;
                packed.native_vnni_blocks_per_row = blocks_per_row;

                VnniPackContext ctx;
                ctx.raw_bytes = reinterpret_cast<const uint8_t *>(weights->raw_data());
                ctx.N = static_cast<int>(N_sz);
                ctx.K = static_cast<int>(K_sz);
                ctx.blocks_per_row = blocks_per_row;
                ctx.payload_bytes = payload_bytes;
                ctx.payload_array = packed.native_vnni_payload.data();
                ctx.scales_array = packed.native_vnni_scales.data();
                ctx.mins_array = nullptr;
                ctx.emins_array = nullptr;

                for (size_t n = 0; n < N_sz; ++n)
                    for (int b = 0; b < blocks_per_row; ++b)
                        weights->packVnniBlock(ctx, static_cast<int>(n), b);
            }

            // Calculate native VNNI weight bytes
            r.weight_bytes = static_cast<double>(packed.native_vnni_payload.size()) +
                             static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t));

            // Set tuning overrides BEFORE creating kernel
            rocmGemv_native_vnni_set_tuning_overrides(kb_forced, target_waves);

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), budget);
            if (!workspace->allocate(reqs))
            {
                rocmGemv_native_vnni_reset_tuning_overrides();
                return r;
            }
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(0)))
            {
                kernel.unbindWorkspace();
                rocmGemv_native_vnni_reset_tuning_overrides();
                return r;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(0)))
            {
                kernel.unbindWorkspace();
                rocmGemv_native_vnni_reset_tuning_overrides();
                return r;
            }

            // Correctness
            {
                kernel.multiply_tensor(input.get(), output.get(), M, shape.N, shape.K);
                (void)hipDeviceSynchronize();
                output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

                if (gpu_w && gpu_w->d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input.get());
                    const float *d_input = reinterpret_cast<const float *>(in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        const size_t out_elems = static_cast<size_t>(shape.N);
                        float *d_ref = nullptr;
                        if (hipMalloc(&d_ref, out_elems * sizeof(float)) == hipSuccess)
                        {
                            if (gpuReferenceFP32Gemm(d_input, gpu_w->d_weights, d_ref, M, shape.N, shape.K, 0))
                            {
                                (void)hipDeviceSynchronize();
                                r.cosine_sim = gpuCosineSimilarity(
                                    reinterpret_cast<const float *>(
                                        dynamic_cast<FP32Tensor *>(output.get())->gpu_data_ptr()),
                                    d_ref, out_elems, 0);
                                r.correctness_pass = (r.cosine_sim >= COSINE_SIM_GATE);
                            }
                            (void)hipFree(d_ref);
                        }
                    }
                }
                output->ensureOnDevice(DeviceId::rocm(0));
            }

            // Timed runs
            auto times = timeKernel(kernel, input.get(), output.get(), shape.N, shape.K);
            computeStats(times, r.mean_us, r.min_us, r.stddev_us);

            r.eff_bw_gbps = (r.weight_bytes / (r.min_us * 1e-6)) / 1e9;
            r.bw_efficiency = (r.eff_bw_gbps / HBM2_PEAK_GBPS) * 100.0;

            kernel.unbindWorkspace();
            rocmGemv_native_vnni_reset_tuning_overrides();
            return r;
        }
#endif
    };

    // =========================================================================
    // Test: Per-shape INT8 vs NativeVNNI baseline comparison
    // =========================================================================

    TEST_F(Q8_0_GEMV_Tuning, INT8_vs_NativeVNNI_Baseline)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  Q8_0 GEMV: INT8-VNNI vs Native-VNNI (Qwen2.5-7B shapes)  ║\n");
        fprintf(stderr, "║  Device: %-49s ║\n", device_name_.c_str());
        fprintf(stderr, "║  %d warmup + %d bench runs per variant                      ║\n",
                WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n\n");

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "N" << "K" << "Path" << "Weight MB"
              << "Min μs" << "Mean μs" << "BW GB/s" << "BW Eff %"
              << "Cosine" << "vs INT8"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(3).set_cell_text_align(fort::text_align::left);
        for (int c : {1, 2, 4, 5, 6, 7, 8, 9, 10})
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &shape : SHAPES_7B)
        {
            fprintf(stderr, "[Shape %s] N=%d K=%d ... ", shape.name.c_str(), shape.N, shape.K);

            // Create FP32 reference weights for correctness
            GpuWeightsCache gpu_w;
            {
                auto tmp = TestTensorFactory::createQ8_0Random(
                    {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
                std::vector<float> fp32(static_cast<size_t>(shape.N) * shape.K);
                tmp->to_fp32(fp32.data());
                gpu_w.upload(fp32.data(), shape.N, shape.K, 0);
            }

            // INT8 baseline
            auto r_int8 = benchmarkINT8Path(shape, &gpu_w);
            fprintf(stderr, "INT8=%.1fμs ", r_int8.min_us);

            // Native VNNI default
            auto r_native = benchmarkNativeVNNIPath(shape, &gpu_w);
            r_native.speedup_vs_int8 = (r_int8.min_us > 0) ? r_int8.min_us / r_native.min_us : 0.0;
            fprintf(stderr, "NV=%.1fμs (%.2fx)\n", r_native.min_us, r_native.speedup_vs_int8);

            auto emit_row = [&](const TuningResult &r)
            {
                char b_n[16], b_k[16], b_mb[16], b_min[16], b_mean[16];
                char b_bw[16], b_eff[16], b_cos[16], b_speedup[16];
                snprintf(b_n, sizeof(b_n), "%d", r.N);
                snprintf(b_k, sizeof(b_k), "%d", r.K);
                snprintf(b_mb, sizeof(b_mb), "%.2f", r.weight_bytes / (1024.0 * 1024.0));
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_mean, sizeof(b_mean), "%.1f", r.mean_us);
                snprintf(b_bw, sizeof(b_bw), "%.1f", r.eff_bw_gbps);
                snprintf(b_eff, sizeof(b_eff), "%.1f%%", r.bw_efficiency);
                snprintf(b_cos, sizeof(b_cos), "%.4f", r.cosine_sim);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx",
                         r.speedup_vs_int8 > 0 ? r.speedup_vs_int8 : 1.0);
                table << r.shape_name << b_n << b_k << r.variant << b_mb
                      << b_min << b_mean << b_bw << b_eff << b_cos << b_speedup
                      << fort::endr;
            };

            r_int8.speedup_vs_int8 = 1.0;
            emit_row(r_int8);
            emit_row(r_native);
            table << fort::separator;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
        fprintf(stderr, "vs INT8: >1.0x = native-VNNI is faster, <1.0x = INT8 is faster\n\n");

        // Print packing overhead summary
        fprintf(stderr, "Packing overhead analysis:\n");
        fprintf(stderr, "  INT8-VNNI:   N×K bytes (INT8 layout) + N×4 bytes (FP32 per-row scale)\n");
        fprintf(stderr, "  Native-VNNI: N×K bytes (payload)     + N×K/32×2 bytes (FP16 per-block scale)\n");
        fprintf(stderr, "  Native-VNNI reads ~6.2%% more data per GEMV (per-block vs per-row scales)\n");
#endif
    }

    // =========================================================================
    // Test: TARGET_WAVES_PER_CU sweep
    // =========================================================================

    TEST_F(Q8_0_GEMV_Tuning, Sweep_TargetWaves)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const std::vector<int> waves_values = {1, 2, 4, 8, 12, 16, 20, 24, 32, 40};

        // Test on the two most common decode shapes
        const std::vector<GEMVShape> sweep_shapes = {
            {"7B_QKV", 3584 * 3, 3584},
            {"7B_FFN_GateUp", 18944, 3584},
            {"7B_FFN_Down", 3584, 18944},
            {"7B_AttnOut", 3584, 3584},
        };

        fprintf(stderr, "\n╔═══════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  Q8_0 Native-VNNI: TARGET_WAVES_PER_CU Sweep                 ║\n");
        fprintf(stderr, "║  Device: %-52s ║\n", device_name_.c_str());
        fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

        for (const auto &shape : sweep_shapes)
        {
            fprintf(stderr, "[Shape %s] N=%d K=%d bpr=%d\n",
                    shape.name.c_str(), shape.N, shape.K, shape.K / 32);

            // INT8 reference
            auto r_int8 = benchmarkINT8Path(shape, nullptr);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Waves/CU" << "Min μs" << "Mean μs" << "σ μs"
                  << "BW GB/s" << "BW Eff %" << "vs INT8" << "Cosine"
                  << fort::endr;
            for (int c = 0; c <= 7; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            // INT8 baseline row
            {
                char b[8][24];
                snprintf(b[0], sizeof(b[0]), "INT8");
                snprintf(b[1], sizeof(b[1]), "%.1f", r_int8.min_us);
                snprintf(b[2], sizeof(b[2]), "%.1f", r_int8.mean_us);
                snprintf(b[3], sizeof(b[3]), "%.1f", r_int8.stddev_us);
                snprintf(b[4], sizeof(b[4]), "%.1f", r_int8.eff_bw_gbps);
                snprintf(b[5], sizeof(b[5]), "%.1f%%", r_int8.bw_efficiency);
                snprintf(b[6], sizeof(b[6]), "1.00x");
                snprintf(b[7], sizeof(b[7]), "%.4f", r_int8.cosine_sim);
                table << b[0] << b[1] << b[2] << b[3] << b[4] << b[5] << b[6] << b[7] << fort::endr;
                table << fort::separator;
            }

            double best_min = 1e9;
            int best_waves = -1;

            for (int tw : waves_values)
            {
                auto r = benchmarkNativeVNNIPath(shape, nullptr, -1, tw);
                double speedup = (r_int8.min_us > 0 && r.min_us > 0) ? r_int8.min_us / r.min_us : 0.0;

                if (r.min_us < best_min)
                {
                    best_min = r.min_us;
                    best_waves = tw;
                }

                char b[8][24];
                snprintf(b[0], sizeof(b[0]), "%d", tw);
                snprintf(b[1], sizeof(b[1]), "%.1f", r.min_us);
                snprintf(b[2], sizeof(b[2]), "%.1f", r.mean_us);
                snprintf(b[3], sizeof(b[3]), "%.1f", r.stddev_us);
                snprintf(b[4], sizeof(b[4]), "%.1f", r.eff_bw_gbps);
                snprintf(b[5], sizeof(b[5]), "%.1f%%", r.bw_efficiency);
                snprintf(b[6], sizeof(b[6]), "%.2fx", speedup);
                snprintf(b[7], sizeof(b[7]), "%.4f", r.cosine_sim);
                table << b[0] << b[1] << b[2] << b[3] << b[4] << b[5] << b[6] << b[7] << fort::endr;
            }

            double best_speedup = (r_int8.min_us > 0) ? r_int8.min_us / best_min : 0.0;
            fprintf(stderr, "  Best: tw=%d → %.1f μs (%.2fx vs INT8 %.1f μs)\n",
                    best_waves, best_min, best_speedup, r_int8.min_us);
            fprintf(stderr, "%s\n\n", table.to_string().c_str());
        }
#endif
    }

    // =========================================================================
    // Test: Direct KB override sweep
    // =========================================================================

    TEST_F(Q8_0_GEMV_Tuning, Sweep_KB)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const std::vector<GEMVShape> sweep_shapes = {
            {"7B_QKV", 3584 * 3, 3584},
            {"7B_FFN_GateUp", 18944, 3584},
            {"7B_FFN_Down", 3584, 18944},
            {"7B_AttnOut", 3584, 3584},
        };

        fprintf(stderr, "\n╔═══════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  Q8_0 Native-VNNI: KB (K-partitions) Sweep                   ║\n");
        fprintf(stderr, "║  Device: %-52s ║\n", device_name_.c_str());
        fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

        for (const auto &shape : sweep_shapes)
        {
            const int blocks_per_row = shape.K / 32;

            // Generate KB values: factors of blocks_per_row + nearby values
            std::vector<int> kb_values = {1};
            for (int kb = 2; kb <= std::min(blocks_per_row, 64); ++kb)
            {
                if (blocks_per_row % kb == 0)
                    kb_values.push_back(kb);
            }
            // Cap at ~15 data points for manageability
            if (kb_values.size() > 15)
            {
                std::vector<int> trimmed = {1};
                size_t step = kb_values.size() / 12;
                for (size_t i = step; i < kb_values.size() - 1; i += step)
                    trimmed.push_back(kb_values[i]);
                trimmed.push_back(kb_values.back());
                kb_values = trimmed;
            }

            fprintf(stderr, "[Shape %s] N=%d K=%d bpr=%d | Testing %zu KB values\n",
                    shape.name.c_str(), shape.N, shape.K, blocks_per_row, kb_values.size());

            // INT8 reference
            auto r_int8 = benchmarkINT8Path(shape, nullptr);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "KB" << "Blocks/Part" << "Min μs" << "Mean μs" << "σ μs"
                  << "BW GB/s" << "BW Eff %" << "vs INT8" << "Cosine"
                  << fort::endr;
            for (int c = 0; c <= 8; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            // INT8 baseline row
            {
                char b[9][24];
                snprintf(b[0], sizeof(b[0]), "INT8");
                snprintf(b[1], sizeof(b[1]), "-");
                snprintf(b[2], sizeof(b[2]), "%.1f", r_int8.min_us);
                snprintf(b[3], sizeof(b[3]), "%.1f", r_int8.mean_us);
                snprintf(b[4], sizeof(b[4]), "%.1f", r_int8.stddev_us);
                snprintf(b[5], sizeof(b[5]), "%.1f", r_int8.eff_bw_gbps);
                snprintf(b[6], sizeof(b[6]), "%.1f%%", r_int8.bw_efficiency);
                snprintf(b[7], sizeof(b[7]), "1.00x");
                snprintf(b[8], sizeof(b[8]), "%.4f", r_int8.cosine_sim);
                table << b[0] << b[1] << b[2] << b[3] << b[4] << b[5] << b[6] << b[7] << b[8] << fort::endr;
                table << fort::separator;
            }

            double best_min = 1e9;
            int best_kb = -1;

            for (int kb : kb_values)
            {
                auto r = benchmarkNativeVNNIPath(shape, nullptr, kb);
                double speedup = (r_int8.min_us > 0 && r.min_us > 0) ? r_int8.min_us / r.min_us : 0.0;

                if (r.min_us > 0 && r.min_us < best_min)
                {
                    best_min = r.min_us;
                    best_kb = kb;
                }

                int blocks_per_part = (kb > 0) ? blocks_per_row / kb : blocks_per_row;

                char b[9][24];
                snprintf(b[0], sizeof(b[0]), "%d", kb);
                snprintf(b[1], sizeof(b[1]), "%d", blocks_per_part);
                snprintf(b[2], sizeof(b[2]), "%.1f", r.min_us);
                snprintf(b[3], sizeof(b[3]), "%.1f", r.mean_us);
                snprintf(b[4], sizeof(b[4]), "%.1f", r.stddev_us);
                snprintf(b[5], sizeof(b[5]), "%.1f", r.eff_bw_gbps);
                snprintf(b[6], sizeof(b[6]), "%.1f%%", r.bw_efficiency);
                snprintf(b[7], sizeof(b[7]), "%.2fx", speedup);
                snprintf(b[8], sizeof(b[8]), "%.4f", r.cosine_sim);
                table << b[0] << b[1] << b[2] << b[3] << b[4] << b[5] << b[6] << b[7] << b[8] << fort::endr;
            }

            double best_speedup = (r_int8.min_us > 0) ? r_int8.min_us / best_min : 0.0;
            fprintf(stderr, "  Best: kb=%d → %.1f μs (%.2fx vs INT8 %.1f μs)\n",
                    best_kb, best_min, best_speedup, r_int8.min_us);
            fprintf(stderr, "%s\n\n", table.to_string().c_str());
        }
#endif
    }

    // =========================================================================
    // Test: Combined TW+KB grid sweep for LM_Head and QKV
    // =========================================================================

    TEST_F(Q8_0_GEMV_Tuning, Sweep_TW_KB_LMHead_QKV)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const std::vector<GEMVShape> focus_shapes = {
            {"7B_QKV", 3584 * 3, 3584},
            {"7B_LM_Head", 152064, 3584},
        };

        const std::vector<int> tw_values = {2, 3, 4, 5, 6, 8};
        const std::vector<int> kb_values_qkv = {1, 2, 4, 7, 8, 14}; // factors of 112
        const std::vector<int> kb_values_lm = {1, 2, 4, 7, 8};      // factors of 112

        fprintf(stderr, "\n╔═══════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  Q8_0 Native-VNNI: Combined TW+KB Sweep (QKV + LM_Head)      ║\n");
        fprintf(stderr, "║  Device: %-52s ║\n", device_name_.c_str());
        fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

        for (const auto &shape : focus_shapes)
        {
            // INT8 reference
            auto r_int8 = benchmarkINT8Path(shape, nullptr);
            fprintf(stderr, "[%s] INT8 baseline: %.1f μs (%.1f GB/s, %.1f%% eff)\n",
                    shape.name.c_str(), r_int8.min_us, r_int8.eff_bw_gbps, r_int8.bw_efficiency);

            // Default native VNNI (with tw=4 heuristic)
            auto r_default = benchmarkNativeVNNIPath(shape, nullptr);
            double def_speedup = (r_int8.min_us > 0) ? r_int8.min_us / r_default.min_us : 0.0;
            fprintf(stderr, "  Default (tw=4 heuristic): %.1f μs (%.2fx vs INT8)\n",
                    r_default.min_us, def_speedup);

            const auto &kb_values = (shape.N > 100000) ? kb_values_lm : kb_values_qkv;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "TW" << "KB" << "Min μs" << "Mean μs" << "σ μs"
                  << "BW GB/s" << "vs INT8" << fort::endr;
            for (int c = 0; c <= 6; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            double best_min = 1e9;
            int best_tw = -1, best_kb = -1;

            for (int tw : tw_values)
            {
                for (int kb : kb_values)
                {
                    auto r = benchmarkNativeVNNIPath(shape, nullptr, kb, tw);
                    double speedup = (r_int8.min_us > 0 && r.min_us > 0) ? r_int8.min_us / r.min_us : 0.0;

                    if (r.min_us > 0 && r.min_us < best_min)
                    {
                        best_min = r.min_us;
                        best_tw = tw;
                        best_kb = kb;
                    }

                    char b[7][24];
                    snprintf(b[0], sizeof(b[0]), "%d", tw);
                    snprintf(b[1], sizeof(b[1]), "%d", kb);
                    snprintf(b[2], sizeof(b[2]), "%.1f", r.min_us);
                    snprintf(b[3], sizeof(b[3]), "%.1f", r.mean_us);
                    snprintf(b[4], sizeof(b[4]), "%.1f", r.stddev_us);
                    snprintf(b[5], sizeof(b[5]), "%.1f", r.eff_bw_gbps);
                    snprintf(b[6], sizeof(b[6]), "%.2fx", speedup);
                    table << b[0] << b[1] << b[2] << b[3] << b[4] << b[5] << b[6] << fort::endr;
                }
                table << fort::separator;
            }

            double best_speedup = (r_int8.min_us > 0) ? r_int8.min_us / best_min : 0.0;
            fprintf(stderr, "  BEST: tw=%d kb=%d → %.1f μs (%.2fx vs INT8 %.1f μs)\n\n",
                    best_tw, best_kb, best_min, best_speedup, r_int8.min_us);
            fprintf(stderr, "%s\n", table.to_string().c_str());
        }
#endif
    }

} // anonymous namespace
