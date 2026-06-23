/**
 * @file Perf__ROCmTPScaling.cpp
 * @brief TP-scaling performance analysis for ROCm GEMV (decode) and GEMM (prefill) kernels
 *
 * Measures kernel performance at full (unsharded) and TP-sharded dimensions
 * for 2-way and 4-way tensor parallelism. Reports per-shape TP scaling
 * efficiency to identify dispatch/tiling regressions at small N or K.
 *
 * Sharding modes (Megatron-style):
 *   - COLUMN_PARALLEL: Q/K/V proj, FFN Gate, FFN Up, LM Head → N divided by TP
 *   - ROW_PARALLEL:    Wo proj, FFN Down                     → K divided by TP
 *
 * Models tested: Qwen2.5-0.5B, Qwen2.5-3B, Qwen2.5-7B
 *
 * @date March 2026
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>

#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

// ============================================================================
// Extern C kernel APIs (same as Perf__ROCmGemvKernel.cpp)
// ============================================================================

extern "C"
{
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size);

    bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8_vnni,
        float *d_C_fp32,
        const float *d_scales_A_blockwise,
        const float *d_scale_B,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int device_id, void *stream);

    bool rocmGemv_int8_scatter_vnni_blockwise(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8_vnni,
        float *d_C_fp32,
        const float *d_scales_A_blockwise,
        const float *d_scales_B,
        const float *d_bias,
        float *d_partial_buf,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        int device_id, void *stream);
}

namespace
{

    // ============================================================================
    // Model dimension constants
    // ============================================================================

    struct ModelDims
    {
        const char *name;
        int hidden;
        int intermediate;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int vocab;
        int num_layers;
    };

    static constexpr ModelDims kQwen05B = {
        "Qwen2.5-0.5B", 896, 4864, 14, 2, 64, 151936, 24};

    static constexpr ModelDims kQwen3B = {
        "Qwen2.5-3B", 2048, 11008, 16, 2, 128, 151936, 36};

    static constexpr ModelDims kQwen7B = {
        "Qwen2.5-7B", 3584, 18944, 28, 4, 128, 152064, 28};

    static constexpr ModelDims kQwen14B = {
        "Qwen2.5-14B", 5120, 13824, 40, 8, 128, 152064, 40};

    static constexpr ModelDims kQwen32B = {
        "Qwen2.5-32B", 5120, 27648, 40, 8, 128, 152064, 64};

    // ============================================================================
    // TP shape generation
    // ============================================================================

    enum class ShardType
    {
        COLUMN, // N divided by TP degree
        ROW,    // K divided by TP degree
        NONE    // No sharding (TP=1 reference)
    };

    struct TPShape
    {
        const char *name;
        int N;
        int K;
        ShardType shard;
    };

    static std::vector<TPShape> getDecodeShapes(const ModelDims &m, int tp_degree)
    {
        const int H = m.hidden;
        const int I = m.intermediate;
        const int kv_dim = m.num_kv_heads * m.head_dim;

        if (tp_degree <= 1)
        {
            return {
                {"Q proj", H, H, ShardType::NONE},
                {"K proj", kv_dim, H, ShardType::NONE},
                {"V proj", kv_dim, H, ShardType::NONE},
                {"Wo proj", H, H, ShardType::NONE},
                {"FFN Gate", I, H, ShardType::NONE},
                {"FFN Up", I, H, ShardType::NONE},
                {"FFN Down", H, I, ShardType::NONE},
            };
        }

        return {
            {"Q proj", H / tp_degree, H, ShardType::COLUMN},
            {"K proj", kv_dim / tp_degree, H, ShardType::COLUMN},
            {"V proj", kv_dim / tp_degree, H, ShardType::COLUMN},
            {"Wo proj", H, H / tp_degree, ShardType::ROW},
            {"FFN Gate", I / tp_degree, H, ShardType::COLUMN},
            {"FFN Up", I / tp_degree, H, ShardType::COLUMN},
            {"FFN Down", H, I / tp_degree, ShardType::ROW},
        };
    }

    static TPShape getLMHeadShape(const ModelDims &m, int tp_degree)
    {
        if (tp_degree <= 1)
            return {"LM Head", m.vocab, m.hidden, ShardType::NONE};
        return {"LM Head", m.vocab / tp_degree, m.hidden, ShardType::COLUMN};
    }

    // ============================================================================
    // Dispatch path classifier (heuristic — matches INT8 VNNI GEMV dispatch)
    // ============================================================================

    static const char *classifyGemvDispatch(int N, int K)
    {
        const bool use_vec4 = ((N % 4) == 0);

        const bool use_wide = (N >= 8 * K) && (N >= 128);
        if (use_wide)
            return "wide";

        // Unified LDS k-reduce path: all vec4-aligned shapes route here
        // (threshold lowered from N>=128 to any vec4 shape)
        const bool lds_kreduce = !use_wide && use_vec4;
        if (lds_kreduce)
            return "LDS-kred";

        if (N >= 256)
            return "square";

        return "fallback";
    }

    // ============================================================================
    // Benchmark result
    // ============================================================================

    struct GemvBenchResult
    {
        double min_us;
        double mean_us;
        double stddev_us;
        double bw_gbps; // effective bandwidth (GB/s) at min_us
        bool success;
    };

    struct GemmBenchResult
    {
        double min_ms;
        double mean_ms;
        double tflops; // at min_ms
        bool success;
    };

    // ============================================================================
    // Utility
    // ============================================================================

    static std::string formatUs(double us)
    {
        char buf[32];
        if (us < 1.0)
            snprintf(buf, sizeof(buf), "%.2f", us);
        else if (us < 100.0)
            snprintf(buf, sizeof(buf), "%.1f", us);
        else
            snprintf(buf, sizeof(buf), "%.0f", us);
        return buf;
    }

    static std::string formatMs(double ms)
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.3f", ms);
        return buf;
    }

    static std::string formatPct(double pct)
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1f%%", pct);
        return buf;
    }

    static std::string formatBW(double gbps)
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.0f", gbps);
        return buf;
    }

    static std::string formatTFLOPS(double tflops)
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.3f", tflops);
        return buf;
    }

    static std::string shardLabel(ShardType s)
    {
        switch (s)
        {
        case ShardType::COLUMN:
            return "col";
        case ShardType::ROW:
            return "row";
        case ShardType::NONE:
            return "-";
        }
        return "?";
    }

    // ============================================================================
    // Test fixture
    // ============================================================================

    class ROCmTPScalingPerfTest : public ::testing::Test
    {
    protected:
        int device_id_ = 0;
        std::string device_name_;
        bool has_device_ = false;

        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            hipError_t err = hipGetDeviceCount(&count);
            has_device_ = (err == hipSuccess && count > 0);
            if (has_device_)
            {
                (void)hipSetDevice(device_id_);
                hipDeviceProp_t props;
                (void)hipGetDeviceProperties(&props, device_id_);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
            }
#endif
        }

        // =========================================================================
        // VNNI weight packing: row-major [K×N] → VNNI [K/4 × N × 4]
        // =========================================================================
        static void packVnniWeights(
            const std::vector<int8_t> &B_int8,
            int N, int K,
            std::vector<int8_t> &out_vnni)
        {
            out_vnni.clear();
            if ((K % 4) != 0)
                return;

            const size_t k_groups = static_cast<size_t>(K) / 4;
            out_vnni.resize(k_groups * static_cast<size_t>(N) * 4);
            for (int n = 0; n < N; ++n)
            {
                for (size_t kg = 0; kg < k_groups; ++kg)
                {
                    const size_t src = (kg * 4) * static_cast<size_t>(N) + static_cast<size_t>(n);
                    const size_t dst = (kg * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4;
                    out_vnni[dst + 0] = B_int8[src + static_cast<size_t>(0) * N];
                    out_vnni[dst + 1] = B_int8[src + static_cast<size_t>(1) * N];
                    out_vnni[dst + 2] = B_int8[src + static_cast<size_t>(2) * N];
                    out_vnni[dst + 3] = B_int8[src + static_cast<size_t>(3) * N];
                }
            }
        }

        // =========================================================================
        // GEMV benchmark (M=1 decode): measures blockwise kernel only
        // =========================================================================
        GemvBenchResult benchmarkGemvBlockwise(
            int N, int K,
            int warmup_runs = 5,
            int bench_runs = 30)
        {
            GemvBenchResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);

            std::vector<float> h_A(K);
            std::vector<int8_t> h_B(static_cast<size_t>(K) * N);
            std::vector<int8_t> h_B_vnni;
            std::vector<float> h_scale(N);

            for (auto &v : h_A)
                v = dist_a(rng);
            for (auto &v : h_B)
                v = static_cast<int8_t>(dist_b(rng));
            for (auto &v : h_scale)
                v = dist_s(rng);

            packVnniWeights(h_B, N, K, h_B_vnni);

            float *d_A = nullptr, *d_scale = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A_blockwise = nullptr;
            float *d_partial = nullptr;

            const int blocks_per_row = K / 32;
            constexpr int MAX_KB = 64;

            (void)hipMalloc(&d_A, K * sizeof(float));
            (void)hipMalloc(&d_scale, N * sizeof(float));
            (void)hipMalloc(&d_C, N * sizeof(float));
            (void)hipMalloc(&d_A_int8, K * sizeof(int8_t));
            (void)hipMalloc(&d_scale_A_blockwise, blocks_per_row * sizeof(float));
            (void)hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * N * sizeof(float));
            (void)hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            (void)hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            (void)hipDeviceSynchronize();

            // Pre-quantize activations (shared across all iterations)
            rocmQuantGemm_quantizeActivationsBlockwise(
                d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
            (void)hipDeviceSynchronize();

            // Determine which kernel path works
            auto run_gemv = [&]() -> bool
            {
                bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, d_B_vnni, d_C, d_scale_A_blockwise, d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                if (!ok)
                {
                    ok = rocmGemv_int8_scatter_vnni_blockwise(
                        d_A_int8, d_B_vnni, d_C, d_scale_A_blockwise, d_scale, nullptr,
                        d_partial, N, K, 1.0f, 0.0f, nullptr, device_id_, nullptr);
                }
                return ok;
            };

            // Warmup
            for (int i = 0; i < warmup_runs; ++i)
            {
                if (!run_gemv())
                {
                    result.success = false;
                    goto cleanup;
                }
            }
            (void)hipDeviceSynchronize();

            // Benchmark with HIP events
            {
                hipEvent_t ev_start, ev_stop;
                (void)hipEventCreate(&ev_start);
                (void)hipEventCreate(&ev_stop);

                std::vector<double> times_us;
                times_us.reserve(bench_runs);

                for (int i = 0; i < bench_runs; ++i)
                {
                    (void)hipDeviceSynchronize();
                    (void)hipEventRecord(ev_start, 0);

                    if (!run_gemv())
                    {
                        result.success = false;
                        (void)hipEventDestroy(ev_start);
                        (void)hipEventDestroy(ev_stop);
                        goto cleanup;
                    }

                    (void)hipEventRecord(ev_stop, 0);
                    (void)hipEventSynchronize(ev_stop);

                    float ms = 0.0f;
                    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
                    times_us.push_back(static_cast<double>(ms) * 1000.0);
                }

                (void)hipEventDestroy(ev_start);
                (void)hipEventDestroy(ev_stop);

                // Compute stats
                std::sort(times_us.begin(), times_us.end());
                double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
                result.mean_us = sum / static_cast<double>(times_us.size());
                result.min_us = times_us.front();

                double sq_sum = 0.0;
                for (double t : times_us)
                    sq_sum += (t - result.mean_us) * (t - result.mean_us);
                result.stddev_us = std::sqrt(sq_sum / static_cast<double>(times_us.size()));

                // Effective bandwidth: read weights [N*K bytes] + activations [K bytes] in min_us
                const double weight_bytes = static_cast<double>(N) * K;
                const double act_bytes = static_cast<double>(K);
                result.bw_gbps = (weight_bytes + act_bytes) / (result.min_us * 1e-6) / 1e9;
                result.success = true;
            }

        cleanup:
            (void)hipFree(d_A);
            (void)hipFree(d_scale);
            (void)hipFree(d_C);
            (void)hipFree(d_A_int8);
            (void)hipFree(d_scale_A_blockwise);
            (void)hipFree(d_partial);
            (void)hipFree(d_B_vnni);

            return result;
#endif
        }

        // =========================================================================
        // GEMM benchmark (prefill): uses full ROCmQuantisedGemmKernel path
        // =========================================================================
        GemmBenchResult benchmarkGemm(
            int M, int N, int K,
            int warmup_runs = 3,
            int bench_runs = 10)
        {
            GemmBenchResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            // Create Q8_0 weights and pack via standard pipeline
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(N), static_cast<size_t>(K)});

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
            {
                result.success = false;
                return result;
            }

            ROCmQuantisedGemmKernel kernel(&packed, device_id_);

            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id_), 1024ULL * 1024ULL * 1024ULL);
            const auto reqs = kernel.getWorkspaceRequirements(M, N, K);
            EXPECT_TRUE(workspace->allocate(reqs));
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});

            // Warmup
            for (int i = 0; i < warmup_runs; ++i)
                kernel.multiply_tensor(input.get(), output.get(), M, N, K);
            (void)hipDeviceSynchronize();

            // Benchmark
            std::vector<double> times_ms;
            times_ms.reserve(bench_runs);

            for (int i = 0; i < bench_runs; ++i)
            {
                (void)hipDeviceSynchronize();
                auto start = std::chrono::high_resolution_clock::now();
                kernel.multiply_tensor(input.get(), output.get(), M, N, K);
                (void)hipDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }

            std::sort(times_ms.begin(), times_ms.end());
            double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
            result.mean_ms = sum / static_cast<double>(times_ms.size());
            result.min_ms = times_ms.front();
            result.tflops = (2.0 * M * N * K) / (result.min_ms * 1e-3) / 1e12;
            result.success = true;

            return result;
#endif
        }

        // =========================================================================
        // TP scaling analysis for GEMV (decode, M=1)
        // =========================================================================
        void runGemvTPScalingAnalysis(
            const ModelDims &model,
            const std::vector<int> &tp_degrees,
            bool include_lm_head = false)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

            // First, benchmark TP=1 (full) shapes
            auto full_shapes = getDecodeShapes(model, 1);
            if (include_lm_head)
                full_shapes.push_back(getLMHeadShape(model, 1));

            struct ShapeBench
            {
                std::string name;
                int N, K;
                GemvBenchResult result;
            };

            std::vector<ShapeBench> full_results;
            for (const auto &s : full_shapes)
            {
                auto r = benchmarkGemvBlockwise(s.N, s.K);
                ASSERT_TRUE(r.success) << s.name << " N=" << s.N << " K=" << s.K;
                full_results.push_back({s.name, s.N, s.K, r});
            }

            // For each TP degree, benchmark sharded shapes and compute efficiency
            for (int tp : tp_degrees)
            {
                auto tp_shapes = getDecodeShapes(model, tp);
                if (include_lm_head)
                    tp_shapes.push_back(getLMHeadShape(model, tp));

                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);
                table << fort::header
                      << "Shape"
                      << "Shard"
                      << "Full N×K"
                      << "TP N×K"
                      << "Dispatch"
                      << "Full(µs)"
                      << "TP(µs)"
                      << "Ideal(µs)"
                      << "Eff%"
                      << "BW(GB/s)"
                      << fort::endr;
                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::center);
                table.column(2).set_cell_text_align(fort::text_align::right);
                table.column(3).set_cell_text_align(fort::text_align::right);
                table.column(4).set_cell_text_align(fort::text_align::left);
                for (int i = 5; i < 10; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                double total_full_us = 0.0;
                double total_tp_us = 0.0;
                double total_ideal_us = 0.0;

                for (size_t i = 0; i < tp_shapes.size(); ++i)
                {
                    const auto &tp_s = tp_shapes[i];

                    // Skip shapes where TP dimension is too small (< 4)
                    if (tp_s.N < 4 || tp_s.K < 4)
                    {
                        table << tp_s.name << shardLabel(tp_s.shard)
                              << (std::to_string(full_results[i].N) + "×" + std::to_string(full_results[i].K))
                              << (std::to_string(tp_s.N) + "×" + std::to_string(tp_s.K))
                              << "skip" << "-" << "-" << "-" << "-" << "-" << fort::endr;
                        continue;
                    }

                    auto tp_r = benchmarkGemvBlockwise(tp_s.N, tp_s.K);
                    ASSERT_TRUE(tp_r.success) << tp_s.name << " TP=" << tp << " N=" << tp_s.N << " K=" << tp_s.K;

                    const double full_us = full_results[i].result.min_us;
                    const double tp_us = tp_r.min_us;
                    const double ideal_us = full_us / static_cast<double>(tp);
                    const double efficiency = (ideal_us / tp_us) * 100.0;

                    total_full_us += full_us;
                    total_tp_us += tp_us;
                    total_ideal_us += ideal_us;

                    const char *dispatch = classifyGemvDispatch(tp_s.N, tp_s.K);
                    const char *full_dispatch = classifyGemvDispatch(full_results[i].N, full_results[i].K);

                    // Flag dispatch path changes
                    std::string dispatch_str = dispatch;
                    if (std::string(dispatch) != std::string(full_dispatch))
                        dispatch_str += " (was " + std::string(full_dispatch) + ")";

                    table << tp_s.name
                          << shardLabel(tp_s.shard)
                          << (std::to_string(full_results[i].N) + "×" + std::to_string(full_results[i].K))
                          << (std::to_string(tp_s.N) + "×" + std::to_string(tp_s.K))
                          << dispatch_str
                          << formatUs(full_us)
                          << formatUs(tp_us)
                          << formatUs(ideal_us)
                          << formatPct(efficiency)
                          << formatBW(tp_r.bw_gbps)
                          << fort::endr;
                }

                // Summary row
                if (total_ideal_us > 0)
                {
                    const double overall_eff = (total_ideal_us / total_tp_us) * 100.0;
                    table << fort::separator;
                    table << "TOTAL (layer)"
                          << ""
                          << ""
                          << ""
                          << ""
                          << formatUs(total_full_us)
                          << formatUs(total_tp_us)
                          << formatUs(total_ideal_us)
                          << formatPct(overall_eff)
                          << ""
                          << fort::endr;
                }

                char header[256];
                snprintf(header, sizeof(header),
                         "\n%s — GEMV Decode TP=%d Scaling Analysis\n",
                         model.name, tp);
                fprintf(stderr, "%s%s\n", header, table.to_string().c_str());
            }
#endif
        }

        // =========================================================================
        // TP scaling analysis for GEMM (prefill)
        // =========================================================================
        void runGemmTPScalingAnalysis(
            const ModelDims &model,
            const std::vector<int> &batch_sizes,
            const std::vector<int> &tp_degrees)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

            // Key GEMM shapes (not all projections — focus on bottlenecks)
            struct GemmShape
            {
                const char *name;
                int N_full;
                int K_full;
                ShardType shard;
            };

            const int H = model.hidden;
            const int I = model.intermediate;

            const std::vector<GemmShape> gemm_shapes = {
                {"Wo proj", H, H, ShardType::ROW},
                {"FFN Gate", I, H, ShardType::COLUMN},
                {"FFN Down", H, I, ShardType::ROW},
            };

            for (int M : batch_sizes)
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);
                table << fort::header
                      << "Shape"
                      << "TP"
                      << "N"
                      << "K"
                      << "Time(ms)"
                      << "TFLOPS"
                      << "Eff%"
                      << fort::endr;
                table.column(0).set_cell_text_align(fort::text_align::left);
                for (int i = 1; i < 7; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (const auto &shape : gemm_shapes)
                {
                    // TP=1 baseline
                    auto base = benchmarkGemm(M, shape.N_full, shape.K_full);
                    if (!base.success)
                        continue;

                    table << shape.name << "1"
                          << shape.N_full << shape.K_full
                          << formatMs(base.min_ms)
                          << formatTFLOPS(base.tflops)
                          << "100.0%"
                          << fort::endr;

                    for (int tp : tp_degrees)
                    {
                        int tp_N = shape.N_full;
                        int tp_K = shape.K_full;

                        if (shape.shard == ShardType::COLUMN)
                            tp_N = shape.N_full / tp;
                        else
                            tp_K = shape.K_full / tp;

                        if (tp_N < 4 || tp_K < 4)
                            continue;

                        auto tp_r = benchmarkGemm(M, tp_N, tp_K);
                        if (!tp_r.success)
                            continue;

                        const double ideal_ms = base.min_ms / static_cast<double>(tp);
                        const double efficiency = (ideal_ms / tp_r.min_ms) * 100.0;

                        table << shape.name << std::to_string(tp)
                              << tp_N << tp_K
                              << formatMs(tp_r.min_ms)
                              << formatTFLOPS(tp_r.tflops)
                              << formatPct(efficiency)
                              << fort::endr;
                    }

                    table << fort::separator;
                }

                char header[256];
                snprintf(header, sizeof(header),
                         "\n%s — GEMM Prefill M=%d TP Scaling\n",
                         model.name, M);
                fprintf(stderr, "%s%s\n", header, table.to_string().c_str());
            }
#endif
        }
    };

    // ============================================================================
    // GEMV Decode TP Scaling Tests
    // ============================================================================

    TEST_F(ROCmTPScalingPerfTest, GEMV_TPScaling_Qwen7B)
    {
        runGemvTPScalingAnalysis(kQwen7B, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMV_TPScaling_Qwen3B)
    {
        runGemvTPScalingAnalysis(kQwen3B, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMV_TPScaling_Qwen05B)
    {
        runGemvTPScalingAnalysis(kQwen05B, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMV_TPScaling_Qwen14B)
    {
        runGemvTPScalingAnalysis(kQwen14B, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMV_TPScaling_Qwen32B)
    {
        runGemvTPScalingAnalysis(kQwen32B, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMV_TPScaling_AllModels_WithLMHead)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        const char *run = std::getenv("LLAMINAR_RUN_TP_SCALING_FULL");
        if (!run || std::string(run) != "1")
            GTEST_SKIP() << "Set LLAMINAR_RUN_TP_SCALING_FULL=1 to run";

        for (const auto *model : {&kQwen05B, &kQwen3B, &kQwen7B, &kQwen14B, &kQwen32B})
        {
            runGemvTPScalingAnalysis(*model, {2, 4}, true);
        }
#endif
    }

    // ============================================================================
    // GEMM Prefill TP Scaling Tests
    // ============================================================================

    TEST_F(ROCmTPScalingPerfTest, GEMM_TPScaling_Qwen7B)
    {
        runGemmTPScalingAnalysis(kQwen7B, {128, 512}, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMM_TPScaling_Qwen3B)
    {
        runGemmTPScalingAnalysis(kQwen3B, {128, 512}, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMM_TPScaling_Qwen05B)
    {
        runGemmTPScalingAnalysis(kQwen05B, {128, 512}, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMM_TPScaling_Qwen14B)
    {
        runGemmTPScalingAnalysis(kQwen14B, {128, 512}, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMM_TPScaling_Qwen32B)
    {
        runGemmTPScalingAnalysis(kQwen32B, {128, 512}, {2, 4});
    }

    TEST_F(ROCmTPScalingPerfTest, GEMM_TPScaling_AllModels_Extended)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        const char *run = std::getenv("LLAMINAR_RUN_TP_SCALING_FULL");
        if (!run || std::string(run) != "1")
            GTEST_SKIP() << "Set LLAMINAR_RUN_TP_SCALING_FULL=1 to run";

        for (const auto *model : {&kQwen05B, &kQwen3B, &kQwen7B, &kQwen14B, &kQwen32B})
        {
            runGemmTPScalingAnalysis(*model, {1, 32, 128, 512, 1024}, {2, 4});
        }
#endif
    }

} // anonymous namespace
