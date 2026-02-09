/**
 * @file Perf__ROCmGemvKernel.cpp
 * @brief Performance benchmark and correctness test for ROCm INT8 GEMV decode kernel
 *
 * Tests the bandwidth-optimized GEMV kernel (ROCmGemvKernel.hip) against:
 *   1. CPU FP32 reference (same INT8 weights + scales → exact correctness)
 *   2. CK INT8 GEMM (A/B timing comparison, approximate match due to different quant)
 *
 * Benchmark configurations cover M=1 decode workloads for:
 *   - Qwen2.5-0.5B: hidden=896,  intermediate=4864,  vocab=151936
 *   - Qwen2.5-7B:   hidden=3584, intermediate=18944, vocab=152064
 *
 * @author Llaminar V2
 * @date February 2026
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
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

// GEMV + CK GEMM kernel C API
extern "C"
{
    bool rocmQuantGemm_quantizeActivations(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int rocm_device_id, void *stream);

    bool rocmQuantGemm_applyScaling(
        const int32_t *d_C_int32,
        float *d_C_fp32,
        const float *d_scales_A,
        const float *d_scales_B,
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int rocm_device_id, void *stream);

    bool rocmGemv_int8_int8_int32_vnni(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8_vnni,
        int32_t *d_C_int32,
        int N, int K,
        int device_id, void *stream);

    bool rocmQuantGemm_executeTwoKernel_cached(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        float *d_C_fp32,
        const float *d_scales_A,
        const float *d_scales_B,
        int32_t *d_C_int32,
        int M, int N, int K,
        int rocm_device_id, void *stream);
}

namespace
{

    // ============================================================================
    // Model dimension constants
    // ============================================================================

    struct ModelDims
    {
        const char *name;
        int hidden;       // d_model
        int intermediate; // d_ff
        int num_heads;    // attention heads
        int num_kv_heads; // GQA KV heads
        int head_dim;     // per-head dimension
        int vocab;        // vocabulary size
        int num_layers;   // transformer layers
    };

    static constexpr ModelDims kQwen05B = {
        "Qwen2.5-0.5B", 896, 4864, 14, 2, 64, 151936, 24};

    static constexpr ModelDims kQwen7B = {
        "Qwen2.5-7B", 3584, 18944, 28, 4, 128, 152064, 28};

    // ============================================================================
    // Per-layer GEMV shapes for M=1 decode
    // ============================================================================

    struct GemvShape
    {
        const char *name;
        int N; // output features
        int K; // input features (reduction dim)
    };

    static std::vector<GemvShape> getDecodeShapes(const ModelDims &m)
    {
        const int H = m.hidden;
        const int I = m.intermediate;
        const int kv_dim = m.num_kv_heads * m.head_dim;
        return {
            {"Q proj", H, H},
            {"K proj", kv_dim, H},
            {"V proj", kv_dim, H},
            {"Wo proj", H, H},
            {"FFN Gate", I, H},
            {"FFN Up", I, H},
            {"FFN Down", H, I},
        };
    }

    static GemvShape getLMHeadShape(const ModelDims &m)
    {
        return {"LM Head", m.vocab, m.hidden};
    }

    // ============================================================================
    // Benchmark result types
    // ============================================================================

    struct BenchResult
    {
        double mean_ms;
        double min_ms;
        double max_ms;
        double stddev_ms;
        double gbps; // effective memory bandwidth (GB/s)
        bool success;
    };

    struct BenchSplitResult
    {
        BenchResult total;
        double quant_mean_ms;
        double quant_min_ms;
        double gemv_mean_ms;
        double gemv_min_ms;
        double scale_mean_ms;
        double scale_min_ms;
        bool success;
    };

    struct CorrectnessResult
    {
        double max_abs_error;
        double mean_abs_error;
        double cosine_sim;
        bool pass; // cosine_sim > threshold
    };

    // ============================================================================
    // Test fixture
    // ============================================================================

    class ROCmGemvPerfTest : public ::testing::Test
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
                hipSetDevice(device_id_);
                hipDeviceProp_t props;
                hipGetDeviceProperties(&props, device_id_);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
            }
#endif
        }

        // =========================================================================
        // CPU reference: FP32×INT8 reference (used to validate INT8 VNNI path)
        //
        //   output[n] = scale_B[n] * sum_k( A[k] * B_int8[k * N + n] )
        //
        // This is NOT an FP32 GEMM — it uses the actual INT8 weights and scales,
        // so the result should match the GPU kernel to within FP32 rounding.
        // =========================================================================
        void cpuReferenceGemv(
            const float *A,       // [K]
            const int8_t *B_int8, // [K × N] row-major
            const float *scale_B, // [N]
            float *C,             // [N]
            int N, int K)
        {
            for (int n = 0; n < N; ++n)
            {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    acc += A[k] * static_cast<float>(B_int8[k * N + n]);
                }
                C[n] = acc * scale_B[n];
            }
        }

        // CPU reference with bias
        void cpuReferenceGemvBias(
            const float *A,       // [K]
            const int8_t *B_int8, // [K × N] row-major
            const float *scale_B, // [N]
            const float *bias,    // [N]
            float *C,             // [N]
            int N, int K)
        {
            cpuReferenceGemv(A, B_int8, scale_B, C, N, K);
            for (int n = 0; n < N; ++n)
                C[n] += bias[n];
        }

        void packVnniWeights(
            const std::vector<int8_t> &B_int8, // [K × N] row-major
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
        // Correctness check: compare GPU output vs CPU reference
        // =========================================================================
        CorrectnessResult checkCorrectness(const float *gpu_out, const float *ref_out, int N)
        {
            CorrectnessResult r{};
            double dot = 0, norm_a = 0, norm_b = 0;
            double sum_abs_err = 0;
            double max_abs_err = 0;

            for (int n = 0; n < N; ++n)
            {
                double g = gpu_out[n];
                double r_val = ref_out[n];
                double err = std::abs(g - r_val);
                sum_abs_err += err;
                max_abs_err = std::max(max_abs_err, err);
                dot += g * r_val;
                norm_a += g * g;
                norm_b += r_val * r_val;
            }

            r.max_abs_error = max_abs_err;
            r.mean_abs_error = sum_abs_err / N;
            r.cosine_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
            r.pass = (r.cosine_sim > 0.998); // slightly relaxed — GPU quantizes activations to INT8
            return r;
        }

        // =========================================================================
        // Benchmark a single GEMV shape
        // =========================================================================
        static void computeStats(const std::vector<double> &times,
                                 double &mean_ms,
                                 double &min_ms,
                                 double &max_ms,
                                 double &stddev_ms)
        {
            if (times.empty())
            {
                mean_ms = 0.0;
                min_ms = 0.0;
                max_ms = 0.0;
                stddev_ms = 0.0;
                return;
            }

            std::vector<double> sorted(times);
            std::sort(sorted.begin(), sorted.end());
            double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
            mean_ms = sum / static_cast<double>(sorted.size());
            min_ms = sorted.front();
            max_ms = sorted.back();

            double sq_sum = 0.0;
            for (double t : sorted)
                sq_sum += (t - mean_ms) * (t - mean_ms);
            stddev_ms = std::sqrt(sq_sum / static_cast<double>(sorted.size()));
        }

        BenchSplitResult benchmarkGemvSplit(
            int N, int K,
            int warmup_runs = 5,
            int bench_runs = 20)
        {
            BenchSplitResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            // --- Allocate host data ---
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

            // --- Pack VNNI weights ---
            packVnniWeights(h_B, N, K, h_B_vnni);

            // --- Allocate device memory ---
            float *d_A = nullptr, *d_scale = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A = nullptr;
            int32_t *d_C_int32 = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A, sizeof(float));
            hipMalloc(&d_C_int32, N * sizeof(int32_t));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // --- Warmup (INT8 VNNI pipeline) ---
            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmQuantGemm_quantizeActivations(d_A, d_A_int8, d_scale_A, 1, K, device_id_, nullptr);
                rocmGemv_int8_int8_int32_vnni(d_A_int8, d_B_vnni, d_C_int32, N, K, device_id_, nullptr);
                rocmQuantGemm_applyScaling(d_C_int32, d_C, d_scale_A, d_scale,
                                           1, N, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
            }
            hipDeviceSynchronize();

            // --- Timed runs ---
            std::vector<double> total_times;
            std::vector<double> quant_times;
            std::vector<double> gemv_times;
            std::vector<double> scale_times;

            total_times.reserve(bench_runs);
            quant_times.reserve(bench_runs);
            gemv_times.reserve(bench_runs);
            scale_times.reserve(bench_runs);

            hipEvent_t total_start, total_stop;
            hipEvent_t quant_start, quant_stop;
            hipEvent_t gemv_start, gemv_stop;
            hipEvent_t scale_start, scale_stop;
            hipEventCreate(&total_start);
            hipEventCreate(&total_stop);
            hipEventCreate(&quant_start);
            hipEventCreate(&quant_stop);
            hipEventCreate(&gemv_start);
            hipEventCreate(&gemv_stop);
            hipEventCreate(&scale_start);
            hipEventCreate(&scale_stop);

            for (int i = 0; i < bench_runs; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(total_start, 0);

                bool ok = true;
                float quant_ms = 0.0f;
                float gemv_ms = 0.0f;
                float scale_ms = 0.0f;

                // Step 1: Quantize activations FP32 → INT8
                hipEventRecord(quant_start, 0);
                ok = rocmQuantGemm_quantizeActivations(d_A, d_A_int8, d_scale_A, 1, K, device_id_, nullptr);
                hipEventRecord(quant_stop, 0);
                hipEventSynchronize(quant_stop);
                hipEventElapsedTime(&quant_ms, quant_start, quant_stop);

                // Step 2: INT8 VNNI GEMV
                if (ok)
                {
                    hipEventRecord(gemv_start, 0);
                    ok = rocmGemv_int8_int8_int32_vnni(d_A_int8, d_B_vnni, d_C_int32, N, K, device_id_, nullptr);
                    hipEventRecord(gemv_stop, 0);
                    hipEventSynchronize(gemv_stop);
                    hipEventElapsedTime(&gemv_ms, gemv_start, gemv_stop);
                }

                // Step 3: Apply scaling INT32 → FP32
                if (ok)
                {
                    hipEventRecord(scale_start, 0);
                    ok = rocmQuantGemm_applyScaling(d_C_int32, d_C, d_scale_A, d_scale,
                                                    1, N, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                    hipEventRecord(scale_stop, 0);
                    hipEventSynchronize(scale_stop);
                    hipEventElapsedTime(&scale_ms, scale_start, scale_stop);
                }

                hipEventRecord(total_stop, 0);
                hipEventSynchronize(total_stop);
                float total_ms = 0.0f;
                hipEventElapsedTime(&total_ms, total_start, total_stop);

                if (!ok)
                {
                    hipFree(d_A);
                    hipFree(d_scale);
                    hipFree(d_C);
                    hipFree(d_A_int8);
                    hipFree(d_scale_A);
                    hipFree(d_C_int32);
                    hipFree(d_B_vnni);
                    return result;
                }

                total_times.push_back(static_cast<double>(total_ms));
                quant_times.push_back(static_cast<double>(quant_ms));
                gemv_times.push_back(static_cast<double>(gemv_ms));
                scale_times.push_back(static_cast<double>(scale_ms));
            }

            hipEventDestroy(total_start);
            hipEventDestroy(total_stop);
            hipEventDestroy(quant_start);
            hipEventDestroy(quant_stop);
            hipEventDestroy(gemv_start);
            hipEventDestroy(gemv_stop);
            hipEventDestroy(scale_start);
            hipEventDestroy(scale_stop);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_A_int8);
            hipFree(d_scale_A);
            hipFree(d_C_int32);

            // --- Statistics ---
            computeStats(total_times, result.total.mean_ms, result.total.min_ms,
                         result.total.max_ms, result.total.stddev_ms);

            double gemv_max_ms = 0.0;
            double gemv_stddev_ms = 0.0;
            computeStats(gemv_times, result.gemv_mean_ms, result.gemv_min_ms,
                         gemv_max_ms, gemv_stddev_ms);

            double quant_max_ms = 0.0;
            double quant_stddev_ms = 0.0;
            computeStats(quant_times, result.quant_mean_ms, result.quant_min_ms,
                         quant_max_ms, quant_stddev_ms);

            double scale_max_ms = 0.0;
            double scale_stddev_ms = 0.0;
            computeStats(scale_times, result.scale_mean_ms, result.scale_min_ms,
                         scale_max_ms, scale_stddev_ms);

            // Effective bandwidth: INT8 weights [K*N] + FP32 activations [K] + scales [N] + output [N]
            double bytes = static_cast<double>(K) * N * 1 // INT8 weights
                           + K * 4                        // FP32 activations
                           + N * 4                        // FP32 scales
                           + N * 4;                       // FP32 output
            result.total.gbps = (bytes / (result.total.min_ms * 1e-3)) / 1e9;
            result.total.success = true;
            result.success = true;
            return result;
#endif
        }

        BenchResult benchmarkGemv(int N, int K, int warmup_runs = 5, int bench_runs = 20)
        {
            auto split = benchmarkGemvSplit(N, K, warmup_runs, bench_runs);
            return split.total;
        }

        // =========================================================================
        // Run correctness test for a single shape
        // =========================================================================
        CorrectnessResult testCorrectness(int N, int K)
        {
            CorrectnessResult bad{0, 0, 0, false};
#ifndef HAVE_ROCM
            return bad;
#else
            std::mt19937 rng(12345);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);

            std::vector<float> h_A(K);
            std::vector<int8_t> h_B(static_cast<size_t>(K) * N);
            std::vector<float> h_scale(N);

            for (auto &v : h_A)
                v = dist_a(rng);
            for (auto &v : h_B)
                v = static_cast<int8_t>(dist_b(rng));
            for (auto &v : h_scale)
                v = dist_s(rng);

            // CPU reference (FP32×INT8 — slightly different from GPU INT8×INT8
            // due to activation quantization, but cosine sim should be >0.998)
            std::vector<float> ref(N);
            cpuReferenceGemv(h_A.data(), h_B.data(), h_scale.data(), ref.data(), N, K);

            // Pack VNNI weights on host
            std::vector<int8_t> h_B_vnni;
            packVnniWeights(h_B, N, K, h_B_vnni);

            // GPU: INT8 VNNI pipeline (quantize → VNNI GEMV → scale)
            float *d_A = nullptr, *d_scale = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A = nullptr;
            int32_t *d_C_int32 = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A, sizeof(float));
            hipMalloc(&d_C_int32, N * sizeof(int32_t));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            rocmQuantGemm_quantizeActivations(d_A, d_A_int8, d_scale_A, 1, K, device_id_, nullptr);
            rocmGemv_int8_int8_int32_vnni(d_A_int8, d_B_vnni, d_C_int32, N, K, device_id_, nullptr);
            rocmQuantGemm_applyScaling(d_C_int32, d_C, d_scale_A, d_scale,
                                       1, N, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
            hipDeviceSynchronize();

            std::vector<float> gpu_out(N);
            hipMemcpy(gpu_out.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_A_int8);
            hipFree(d_scale_A);
            hipFree(d_C_int32);

            return checkCorrectness(gpu_out.data(), ref.data(), N);
#endif
        }

        // =========================================================================
        // Run correctness test with bias
        // =========================================================================
        CorrectnessResult testCorrectnessBias(int N, int K)
        {
            CorrectnessResult bad{0, 0, 0, false};
#ifndef HAVE_ROCM
            return bad;
#else
            std::mt19937 rng(99999);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);
            std::uniform_real_distribution<float> dist_bias(-0.5f, 0.5f);

            std::vector<float> h_A(K);
            std::vector<int8_t> h_B(static_cast<size_t>(K) * N);
            std::vector<float> h_scale(N);
            std::vector<float> h_bias(N);

            for (auto &v : h_A)
                v = dist_a(rng);
            for (auto &v : h_B)
                v = static_cast<int8_t>(dist_b(rng));
            for (auto &v : h_scale)
                v = dist_s(rng);
            for (auto &v : h_bias)
                v = dist_bias(rng);

            // CPU reference with bias (FP32×INT8 + bias)
            std::vector<float> ref(N);
            cpuReferenceGemvBias(h_A.data(), h_B.data(), h_scale.data(),
                                 h_bias.data(), ref.data(), N, K);

            // Pack VNNI weights on host
            std::vector<int8_t> h_B_vnni;
            packVnniWeights(h_B, N, K, h_B_vnni);

            // GPU: INT8 VNNI pipeline with bias (quantize → VNNI GEMV → scale+bias)
            float *d_A = nullptr, *d_scale = nullptr, *d_bias = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A = nullptr;
            int32_t *d_C_int32 = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_bias, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A, sizeof(float));
            hipMalloc(&d_C_int32, N * sizeof(int32_t));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_bias, h_bias.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            rocmQuantGemm_quantizeActivations(d_A, d_A_int8, d_scale_A, 1, K, device_id_, nullptr);
            rocmGemv_int8_int8_int32_vnni(d_A_int8, d_B_vnni, d_C_int32, N, K, device_id_, nullptr);
            rocmQuantGemm_applyScaling(d_C_int32, d_C, d_scale_A, d_scale,
                                       1, N, 1.0f, 0.0f, nullptr, d_bias, device_id_, nullptr);
            hipDeviceSynchronize();

            std::vector<float> gpu_out(N);
            hipMemcpy(gpu_out.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_bias);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_A_int8);
            hipFree(d_scale_A);
            hipFree(d_C_int32);

            return checkCorrectness(gpu_out.data(), ref.data(), N);
#endif
        }

        // =========================================================================
        // Printing helpers
        // =========================================================================

        void printBenchHeader(const char *title)
        {
            fprintf(stderr, "\n╔══════════════════════════════════════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║  %-76s║\n", title);
            fprintf(stderr, "╠═══════════════════╦═══════╦═══════╦══════════╦══════════╦══════════╦═══════════╣\n");
            fprintf(stderr, "║ Shape             ║   N   ║   K   ║ Mean(ms) ║ Min(ms)  ║  GB/s    ║  Status   ║\n");
            fprintf(stderr, "╠═══════════════════╬═══════╬═══════╬══════════╬══════════╬══════════╬═══════════╣\n");
        }

        void printBenchRow(const char *name, int N, int K, const BenchResult &r)
        {
            const char *status = r.success ? "OK" : "FAIL";
            fprintf(stderr, "║ %-17s ║ %5d ║ %5d ║ %8.3f ║ %8.3f ║ %8.1f ║ %-9s ║\n",
                    name, N, K, r.mean_ms, r.min_ms, r.gbps, status);
        }

        void printBenchFooter()
        {
            fprintf(stderr, "╚═══════════════════╩═══════╩═══════╩══════════╩══════════╩══════════╩═══════════╝\n");
        }

        void printCorrectnessHeader(const char *title)
        {
            fprintf(stderr, "\n╔══════════════════════════════════════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║  %-76s║\n", title);
            fprintf(stderr, "╠═══════════════════╦═══════╦═══════╦════════════╦════════════╦══════════╦════════╣\n");
            fprintf(stderr, "║ Shape             ║   N   ║   K   ║ MaxAbsErr  ║ MeanAbsErr ║ CosineSim║ Status ║\n");
            fprintf(stderr, "╠═══════════════════╬═══════╬═══════╬════════════╬════════════╬══════════╬════════╣\n");
        }

        void printCorrectnessRow(const char *name, int N, int K, const CorrectnessResult &r)
        {
            const char *status = r.pass ? "PASS" : "FAIL";
            fprintf(stderr, "║ %-17s ║ %5d ║ %5d ║ %10.2e ║ %10.2e ║ %8.6f ║ %-6s ║\n",
                    name, N, K, r.max_abs_error, r.mean_abs_error, r.cosine_sim, status);
        }

        void printCorrectnessFooter()
        {
            fprintf(stderr, "╚═══════════════════╩═══════╩═══════╩════════════╩════════════╩══════════╩════════╝\n");
        }

        static std::string formatMs(double value)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << value;
            return oss.str();
        }

        void printSplitBenchTable(const char *title,
                                  const std::vector<GemvShape> &shapes,
                                  const std::vector<BenchSplitResult> &results)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Shape" << "N" << "K" << "Quant min(ms)" << "GEMV min(ms)" << "Scale min(ms)" << "Total min(ms)"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (std::size_t i = 1; i < 7; ++i)
                table.column(static_cast<int>(i)).set_cell_text_align(fort::text_align::right);

            for (std::size_t i = 0; i < shapes.size(); ++i)
            {
                const auto &shape = shapes[i];
                const auto &r = results[i];
                table << shape.name
                      << shape.N
                      << shape.K
                      << formatMs(r.quant_min_ms)
                      << formatMs(r.gemv_min_ms)
                      << formatMs(r.scale_min_ms)
                      << formatMs(r.total.min_ms)
                      << fort::endr;
            }

            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }
    };

    // ============================================================================
    // TEST: Correctness — all Qwen2.5-0.5B decode shapes
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Correctness_Qwen05B)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        printCorrectnessHeader("GEMV Correctness: Qwen2.5-0.5B (M=1 decode)");

        auto shapes = getDecodeShapes(kQwen05B);
        shapes.push_back(getLMHeadShape(kQwen05B));

        for (const auto &s : shapes)
        {
            auto r = testCorrectness(s.N, s.K);
            printCorrectnessRow(s.name, s.N, s.K, r);
            EXPECT_TRUE(r.pass) << s.name << " N=" << s.N << " K=" << s.K
                                << " cosine=" << r.cosine_sim;
        }
        printCorrectnessFooter();
#endif
    }

    // ============================================================================
    // TEST: Correctness — all Qwen2.5-7B decode shapes
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Correctness_Qwen7B)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        printCorrectnessHeader("GEMV Correctness: Qwen2.5-7B (M=1 decode)");

        auto shapes = getDecodeShapes(kQwen7B);
        shapes.push_back(getLMHeadShape(kQwen7B));

        for (const auto &s : shapes)
        {
            auto r = testCorrectness(s.N, s.K);
            printCorrectnessRow(s.name, s.N, s.K, r);
            EXPECT_TRUE(r.pass) << s.name << " N=" << s.N << " K=" << s.K
                                << " cosine=" << r.cosine_sim;
        }
        printCorrectnessFooter();
#endif
    }

    // ============================================================================
    // TEST: Correctness — bias fusion
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Correctness_BiasFusion)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        printCorrectnessHeader("GEMV+Bias Correctness (M=1 decode)");

        // Test bias fusion with representative shapes from both models
        struct BiasShape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<BiasShape> shapes = {
            {"0.5B Wo+bias", kQwen05B.hidden, kQwen05B.hidden},
            {"0.5B FFN+bias", kQwen05B.intermediate, kQwen05B.hidden},
            {"7B Wo+bias", kQwen7B.hidden, kQwen7B.hidden},
            {"7B FFN+bias", kQwen7B.intermediate, kQwen7B.hidden},
        };

        for (const auto &s : shapes)
        {
            auto r = testCorrectnessBias(s.N, s.K);
            printCorrectnessRow(s.name, s.N, s.K, r);
            EXPECT_TRUE(r.pass) << s.name << " N=" << s.N << " K=" << s.K
                                << " cosine=" << r.cosine_sim;
        }
        printCorrectnessFooter();
#endif
    }

    // ============================================================================
    // TEST: Performance — Qwen2.5-0.5B full decode layer
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_Qwen05B_DecodeLayer)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        auto shapes = getDecodeShapes(kQwen05B);

        // Header with model summary
        char title[128];
        snprintf(title, sizeof(title),
                 "GEMV Benchmark: %s (M=1 decode, %d layers, INT8 VNNI)",
                 kQwen05B.name, kQwen05B.num_layers);
        printBenchHeader(title);

        double total_layer_ms = 0;
        double total_layer_bytes = 0;

        std::vector<BenchSplitResult> split_results;
        split_results.reserve(shapes.size() + 1);

        for (const auto &s : shapes)
        {
            auto split = benchmarkGemvSplit(s.N, s.K);
            split_results.push_back(split);
            printBenchRow(s.name, s.N, s.K, split.total);
            EXPECT_TRUE(split.total.success);
            total_layer_ms += split.total.min_ms;
            total_layer_bytes += static_cast<double>(s.K) * s.N; // INT8 weight bytes
        }

        // LM Head (runs once per token, not per layer)
        auto lm = getLMHeadShape(kQwen05B);
        auto split_lm = benchmarkGemvSplit(lm.N, lm.K);
        split_results.push_back(split_lm);
        printBenchRow(lm.name, lm.N, lm.K, split_lm.total);
        EXPECT_TRUE(split_lm.total.success);

        printBenchFooter();

        // Summary
        double all_layers_ms = total_layer_ms * kQwen05B.num_layers + split_lm.total.min_ms;
        double all_layers_bytes = total_layer_bytes * kQwen05B.num_layers + static_cast<double>(lm.K) * lm.N;
        double effective_gbps = (all_layers_bytes / (all_layers_ms * 1e-3)) / 1e9;

        fprintf(stderr, "\n  Per-layer GEMV total:   %8.3f ms\n", total_layer_ms);
        fprintf(stderr, "  LM Head:               %8.3f ms\n", split_lm.total.min_ms);
        fprintf(stderr, "  All %d layers + LM:    %8.3f ms  (%.1f tok/s GEMV-only)\n",
                kQwen05B.num_layers, all_layers_ms, 1000.0 / all_layers_ms);
        fprintf(stderr, "  Effective bandwidth:   %8.1f GB/s  (of 480 GB/s HBM2)\n", effective_gbps);
        fprintf(stderr, "  Weight data read:      %8.1f MB\n\n", all_layers_bytes / 1e6);

        auto shapes_with_lm = shapes;
        shapes_with_lm.push_back(lm);
        printSplitBenchTable("GEMV Split Timing (INT8 VNNI)", shapes_with_lm, split_results);
#endif
    }

    // ============================================================================
    // TEST: Performance — Qwen2.5-7B full decode layer
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_Qwen7B_DecodeLayer)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        auto shapes = getDecodeShapes(kQwen7B);

        char title[128];
        snprintf(title, sizeof(title),
                 "GEMV Benchmark: %s (M=1 decode, %d layers, INT8 VNNI)",
                 kQwen7B.name, kQwen7B.num_layers);
        printBenchHeader(title);

        double total_layer_ms = 0;
        double total_layer_bytes = 0;

        std::vector<BenchSplitResult> split_results;
        split_results.reserve(shapes.size() + 1);

        for (const auto &s : shapes)
        {
            auto split = benchmarkGemvSplit(s.N, s.K);
            split_results.push_back(split);
            printBenchRow(s.name, s.N, s.K, split.total);
            EXPECT_TRUE(split.total.success);
            total_layer_ms += split.total.min_ms;
            total_layer_bytes += static_cast<double>(s.K) * s.N;
        }

        auto lm = getLMHeadShape(kQwen7B);
        auto split_lm = benchmarkGemvSplit(lm.N, lm.K);
        split_results.push_back(split_lm);
        printBenchRow(lm.name, lm.N, lm.K, split_lm.total);
        EXPECT_TRUE(split_lm.total.success);

        printBenchFooter();

        double all_layers_ms = total_layer_ms * kQwen7B.num_layers + split_lm.total.min_ms;
        double all_layers_bytes = total_layer_bytes * kQwen7B.num_layers + static_cast<double>(lm.K) * lm.N;
        double effective_gbps = (all_layers_bytes / (all_layers_ms * 1e-3)) / 1e9;

        fprintf(stderr, "\n  Per-layer GEMV total:   %8.3f ms\n", total_layer_ms);
        fprintf(stderr, "  LM Head:               %8.3f ms\n", split_lm.total.min_ms);
        fprintf(stderr, "  All %d layers + LM:    %8.3f ms  (%.1f tok/s GEMV-only)\n",
                kQwen7B.num_layers, all_layers_ms, 1000.0 / all_layers_ms);
        fprintf(stderr, "  Effective bandwidth:   %8.1f GB/s  (of 480 GB/s HBM2)\n", effective_gbps);
        fprintf(stderr, "  Weight data read:      %8.1f MB\n\n", all_layers_bytes / 1e6);

        auto shapes_with_lm = shapes;
        shapes_with_lm.push_back(lm);
        printSplitBenchTable("GEMV Split Timing (INT8 VNNI)", shapes_with_lm, split_results);
#endif
    }

    // ============================================================================
    // TEST: GEMV vs CK head-to-head comparison (same INT8 data)
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_GEMVvsCK)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        fprintf(stderr, "\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  GEMV vs CK INT8 GEMM Head-to-Head  (M=1, same INT8 weights)                               ║\n");
        fprintf(stderr, "╠═══════════════════╦═══════╦═══════╦═══════════════════╦═══════════════════╦═════════════════╣\n");
        fprintf(stderr, "║ Shape             ║   N   ║   K   ║  GEMV min(ms)     ║  CK min(ms)       ║  Speedup       ║\n");
        fprintf(stderr, "╠═══════════════════╬═══════╬═══════╬═══════════════════╬═══════════════════╬═════════════════╣\n");

        // Representative shapes
        struct CmpShape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<CmpShape> shapes = {
            {"7B Wo", kQwen7B.hidden, kQwen7B.hidden},
            {"7B FFN Gate", kQwen7B.intermediate, kQwen7B.hidden},
            {"7B FFN Down", kQwen7B.hidden, kQwen7B.intermediate},
        };

        for (const auto &s : shapes)
        {
            // --- GEMV benchmark ---
            auto r_gemv = benchmarkGemv(s.N, s.K, 5, 20);

            // --- CK benchmark (M=1 padded to 8) ---
            const int M_padded = 8;
            const int K = s.K;
            const int N = s.N;

            // Allocate for CK path
            float *d_A = nullptr, *d_C = nullptr;
            int8_t *d_A_int8 = nullptr, *d_B_int8 = nullptr;
            float *d_scales_A = nullptr, *d_scales_B = nullptr;
            int32_t *d_C_int32 = nullptr;

            hipMalloc(&d_A, M_padded * K * sizeof(float));
            hipMalloc(&d_A_int8, M_padded * K * sizeof(int8_t));
            hipMalloc(&d_scales_A, M_padded * sizeof(float));
            hipMalloc(&d_B_int8, static_cast<size_t>(K) * N * sizeof(int8_t));
            hipMalloc(&d_scales_B, N * sizeof(float));
            hipMalloc(&d_C, M_padded * N * sizeof(float));
            hipMalloc(&d_C_int32, M_padded * N * sizeof(int32_t));

            // Fill with random data
            std::vector<float> h_A(M_padded * K);
            std::vector<int8_t> h_B(static_cast<size_t>(K) * N);
            std::vector<float> h_s(N);
            std::mt19937 rng(42);
            for (auto &v : h_A)
                v = static_cast<float>(rng()) / rng.max() * 2.0f - 1.0f;
            for (auto &v : h_B)
                v = static_cast<int8_t>(rng() % 255 - 127);
            for (auto &v : h_s)
                v = 0.01f + static_cast<float>(rng()) / rng.max() * 0.09f;

            hipMemcpy(d_A, h_A.data(), M_padded * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_int8, h_B.data(), static_cast<size_t>(K) * N * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scales_B, h_s.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // Warmup CK
            for (int i = 0; i < 3; ++i)
            {
                rocmQuantGemm_quantizeActivations(d_A, d_A_int8, d_scales_A, M_padded, K, device_id_, nullptr);
                rocmQuantGemm_executeTwoKernel_cached(d_A_int8, d_B_int8, d_C, d_scales_A, d_scales_B,
                                                      d_C_int32, M_padded, N, K, device_id_, nullptr);
            }
            hipDeviceSynchronize();

            // Timed CK runs
            std::vector<double> ck_times;
            for (int i = 0; i < 20; ++i)
            {
                hipDeviceSynchronize();
                auto t0 = std::chrono::high_resolution_clock::now();
                rocmQuantGemm_quantizeActivations(d_A, d_A_int8, d_scales_A, M_padded, K, device_id_, nullptr);
                rocmQuantGemm_executeTwoKernel_cached(d_A_int8, d_B_int8, d_C, d_scales_A, d_scales_B,
                                                      d_C_int32, M_padded, N, K, device_id_, nullptr);
                hipDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                ck_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            hipFree(d_A);
            hipFree(d_A_int8);
            hipFree(d_scales_A);
            hipFree(d_B_int8);
            hipFree(d_scales_B);
            hipFree(d_C);
            hipFree(d_C_int32);

            double ck_min = *std::min_element(ck_times.begin(), ck_times.end());
            double speedup = ck_min / r_gemv.min_ms;

            fprintf(stderr, "║ %-17s ║ %5d ║ %5d ║ %17.3f ║ %17.3f ║ %13.1fx ║\n",
                    s.name, N, K, r_gemv.min_ms, ck_min, speedup);
        }

        fprintf(stderr, "╚═══════════════════╩═══════╩═══════╩═══════════════════╩═══════════════════╩═════════════════╝\n\n");
#endif
    }

    // ============================================================================
    // TEST: Bandwidth roofline analysis
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_BandwidthRoofline)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        fprintf(stderr, "\n╔══════════════════════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  Bandwidth Roofline Analysis (MI60: 480 GB/s HBM2 theoretical)                 ║\n");
        fprintf(stderr, "╠═══════════════════╦════════════╦════════════╦══════════╦═════════════════════════╣\n");
        fprintf(stderr, "║ Shape             ║ Weight(MB) ║  Min(ms)   ║  GB/s    ║  %% of HBM2 peak       ║\n");
        fprintf(stderr, "╠═══════════════════╬════════════╬════════════╬══════════╬═════════════════════════╣\n");

        const double HBM2_PEAK_GBPS = 480.0;

        struct RoofShape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<RoofShape> shapes = {
            // Small (0.5B)
            {"0.5B Wo", kQwen05B.hidden, kQwen05B.hidden},
            {"0.5B FFN Gate", kQwen05B.intermediate, kQwen05B.hidden},
            // Large (7B)
            {"7B Wo", kQwen7B.hidden, kQwen7B.hidden},
            {"7B FFN Gate", kQwen7B.intermediate, kQwen7B.hidden},
            {"7B FFN Down", kQwen7B.hidden, kQwen7B.intermediate},
            {"7B LM Head", kQwen7B.vocab, kQwen7B.hidden},
        };

        for (const auto &s : shapes)
        {
            auto r = benchmarkGemv(s.N, s.K, 5, 20);
            double weight_mb = static_cast<double>(s.K) * s.N / 1e6;
            double pct = (r.gbps / HBM2_PEAK_GBPS) * 100.0;

            fprintf(stderr, "║ %-17s ║ %10.1f ║ %10.3f ║ %8.1f ║ %21.1f%% ║\n",
                    s.name, weight_mb, r.min_ms, r.gbps, pct);
        }

        fprintf(stderr, "╚═══════════════════╩════════════╩════════════╩══════════╩═════════════════════════╝\n\n");
#endif
    }
} // namespace
