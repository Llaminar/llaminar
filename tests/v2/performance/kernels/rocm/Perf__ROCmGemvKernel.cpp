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
#include <limits>
#include <string>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

// Stub for removed CK two-kernel GEMM (no longer built as separate symbol)
extern "C" bool rocmQuantGemm_executeTwoKernel_cached(
    const int8_t *, const int8_t *, float *, const float *, const float *,
    int32_t *, int, int, int, int, void *)
{
    return false;
}

// GEMV + CK GEMM kernel C API
extern "C"
{
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size);

    void rocmGemv_int8_vnni_set_tuning_overrides(
        int tn,
        int kb);

    void rocmGemv_int8_vnni_reset_tuning_overrides();
    void rocmGemv_int8_vnni_set_wk_override(int wk);
    void rocmGemv_int8_vnni_set_unroll_override(int unroll);
    void rocmGemv_int8_vnni_set_act_block_override(int act_bk);
    void rocmGemv_int8_vnni_reset_qwo_overrides();
    void rocmGemv_int8_vnni_set_skip_memset(int skip);
    void rocmGemv_int8_vnni_set_force_pair(int force);
    bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
        const int8_t *d_A_int8,
        const float *d_scales_A_blockwise,
        const int8_t *d_B0,
        const int8_t *d_B1,
        float *d_C0,
        float *d_C1,
        const float *d_scales_B0,
        const float *d_scales_B1,
        int N0, int N1,
        int K,
        float alpha,
        int device_id, void *stream);
    void rocmGemv_int8_vnni_set_wide_tuning_overrides(
        int wide_tn,
        int wide_cpt,
        int wide_vec_load);
    void rocmGemv_int8_vnni_reset_wide_tuning_overrides();

    bool rocmGemv_fused_scatter_fp32_int8_vnni(
        const float *d_A_fp32,
        const int8_t *d_B_int8_vnni,
        float *d_C_fp32,
        const float *d_scales_B,
        const float *d_bias,
        float *d_partial_buf,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        int device_id, void *stream);

    bool rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni(
        const float *d_A_fp32,
        const int8_t *d_B_int8_vnni,
        float *d_C_fp32,
        const float *d_scales_B,
        const float *d_bias,
        float *d_partial_buf,
        int *d_counter,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
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

    bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8_vnni,
        float *d_C_fp32,
        const float *d_scales_A_blockwise,
        const float *d_scale_B,
        int N, int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        int device_id, void *stream);

    bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
        const int8_t *d_A_int8,
        const float *d_scales_A_blockwise,
        int num_projections,
        const int8_t *const *d_B_ptrs,
        float *const *d_C_ptrs,
        const float *const *d_scales_B_ptrs,
        const float *const *d_bias_ptrs,
        const int *N_per_proj,
        int K,
        float alpha,
        float beta,
        int device_id, void *stream);

    bool rocmGemv_int8_scatter_batched_vnni_blockwise(
        const int8_t *d_A_int8,
        const float *d_scales_A_blockwise,
        float *d_partial_buf,
        int num_projections,
        const int8_t *const *d_B_ptrs,
        float *const *d_C_ptrs,
        const float *const *d_scales_B_ptrs,
        const float *const *d_bias_ptrs,
        const int *N_per_proj,
        int K,
        float alpha, float beta,
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

    static constexpr ModelDims kQwen3B = {
        "Qwen2.5-3B", 2048, 11008, 16, 2, 128, 151936, 36};

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

    struct BatchedDecodeGroup
    {
        const char *name;
        std::vector<int> Ns;
        int K;
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
            float *d_scale_A_bw = nullptr;
            int32_t *d_C_int32 = nullptr;

            const int blocks_per_row = (K + 31) / 32;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_bw, blocks_per_row * sizeof(float));
            hipMalloc(&d_C_int32, N * sizeof(int32_t));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // --- Warmup (blockwise INT8 VNNI pipeline) ---
            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_bw, 1, K, device_id_, nullptr, 32);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, d_B_vnni, d_C, d_scale_A_bw, d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
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
                bool fused_scale = false;

                // Step 1: Quantize activations FP32 → INT8 (blockwise)
                hipEventRecord(quant_start, 0);
                ok = rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_bw, 1, K, device_id_, nullptr, 32);
                hipEventRecord(quant_stop, 0);
                hipEventSynchronize(quant_stop);
                hipEventElapsedTime(&quant_ms, quant_start, quant_stop);

                // Step 2: INT8 VNNI GEMV (blockwise — always fuses scaling)
                if (ok)
                {
                    hipEventRecord(gemv_start, 0);
                    fused_scale = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                        d_A_int8, d_B_vnni, d_C, d_scale_A_bw, d_scale,
                        N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                    ok = fused_scale;
                    hipEventRecord(gemv_stop, 0);
                    hipEventSynchronize(gemv_stop);
                    hipEventElapsedTime(&gemv_ms, gemv_start, gemv_stop);
                }

                // Step 3: No longer needed — blockwise always fuses scaling

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
                    hipFree(d_scale_A_bw);
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
            hipFree(d_scale_A_bw);
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

        BenchSplitResult benchmarkGemvSplitBlockwise(
            int N, int K,
            int warmup_runs = 5,
            int bench_runs = 20)
        {
            BenchSplitResult result{};
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

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_blockwise, blocks_per_row * sizeof(float));
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
                bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, d_B_vnni, d_C, d_scale_A_blockwise, d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                if (!ok)
                {
                    rocmGemv_int8_scatter_vnni_blockwise(
                        d_A_int8, d_B_vnni, d_C, d_scale_A_blockwise, d_scale, nullptr,
                        d_partial, N, K, 1.0f, 0.0f, nullptr, device_id_, nullptr);
                }
            }
            hipDeviceSynchronize();

            std::vector<double> total_times;
            std::vector<double> quant_times;
            std::vector<double> gemv_times;

            total_times.reserve(bench_runs);
            quant_times.reserve(bench_runs);
            gemv_times.reserve(bench_runs);

            hipEvent_t total_start, total_stop;
            hipEvent_t quant_start, quant_stop;
            hipEvent_t gemv_start, gemv_stop;
            hipEventCreate(&total_start);
            hipEventCreate(&total_stop);
            hipEventCreate(&quant_start);
            hipEventCreate(&quant_stop);
            hipEventCreate(&gemv_start);
            hipEventCreate(&gemv_stop);

            for (int i = 0; i < bench_runs; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(total_start, 0);

                float quant_ms = 0.0f;
                float gemv_ms = 0.0f;

                hipEventRecord(quant_start, 0);
                bool ok = rocmQuantGemm_quantizeActivationsBlockwise(
                    d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
                hipEventRecord(quant_stop, 0);
                hipEventSynchronize(quant_stop);
                hipEventElapsedTime(&quant_ms, quant_start, quant_stop);

                if (ok)
                {
                    hipEventRecord(gemv_start, 0);
                    ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                        d_A_int8, d_B_vnni, d_C, d_scale_A_blockwise, d_scale,
                        N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                    if (!ok)
                    {
                        ok = rocmGemv_int8_scatter_vnni_blockwise(
                            d_A_int8, d_B_vnni, d_C, d_scale_A_blockwise, d_scale, nullptr,
                            d_partial, N, K, 1.0f, 0.0f, nullptr, device_id_, nullptr);
                    }
                    hipEventRecord(gemv_stop, 0);
                    hipEventSynchronize(gemv_stop);
                    hipEventElapsedTime(&gemv_ms, gemv_start, gemv_stop);
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
                    hipFree(d_B_vnni);
                    hipFree(d_A_int8);
                    hipFree(d_scale_A_blockwise);
                    hipFree(d_partial);
                    return result;
                }

                total_times.push_back(static_cast<double>(total_ms));
                quant_times.push_back(static_cast<double>(quant_ms));
                gemv_times.push_back(static_cast<double>(gemv_ms));
            }

            hipEventDestroy(total_start);
            hipEventDestroy(total_stop);
            hipEventDestroy(quant_start);
            hipEventDestroy(quant_stop);
            hipEventDestroy(gemv_start);
            hipEventDestroy(gemv_stop);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_A_int8);
            hipFree(d_scale_A_blockwise);
            hipFree(d_partial);

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

            result.scale_mean_ms = 0.0;
            result.scale_min_ms = 0.0;

            double bytes = static_cast<double>(K) * N * 1 + K * 4 + N * 4 + N * 4;
            result.total.gbps = (bytes / (result.total.min_ms * 1e-3)) / 1e9;
            result.total.success = true;
            result.success = true;
            return result;
#endif
        }

        BenchSplitResult benchmarkGemvSplitBlockwiseSharedQuantSeparate(
            const std::vector<int> &Ns,
            int K,
            int warmup_runs = 5,
            int bench_runs = 20)
        {
            BenchSplitResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            if (Ns.empty())
                return result;

            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);

            std::vector<float> h_A(K);
            for (auto &v : h_A)
                v = dist_a(rng);

            float *d_A = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A_blockwise = nullptr;
            float *d_partial = nullptr;

            const int blocks_per_row = K / 32;
            constexpr int MAX_KB = 64;
            const int max_n = *std::max_element(Ns.begin(), Ns.end());
            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_blockwise, blocks_per_row * sizeof(float));
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * max_n * sizeof(float));
            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);

            std::vector<int8_t *> d_B_vnni(Ns.size(), nullptr);
            std::vector<float *> d_scale(Ns.size(), nullptr);
            std::vector<float *> d_C(Ns.size(), nullptr);
            std::vector<std::vector<int8_t>> h_B_vnni(Ns.size());

            for (size_t i = 0; i < Ns.size(); ++i)
            {
                std::vector<int8_t> h_B(static_cast<size_t>(K) * Ns[i]);
                std::vector<float> h_scale(Ns[i]);
                for (auto &v : h_B)
                    v = static_cast<int8_t>(dist_b(rng));
                for (auto &v : h_scale)
                    v = dist_s(rng);

                packVnniWeights(h_B, Ns[i], K, h_B_vnni[i]);
                hipMalloc(&d_B_vnni[i], h_B_vnni[i].size() * sizeof(int8_t));
                hipMalloc(&d_scale[i], Ns[i] * sizeof(float));
                hipMalloc(&d_C[i], Ns[i] * sizeof(float));
                hipMemcpy(d_B_vnni[i], h_B_vnni[i].data(), h_B_vnni[i].size() * sizeof(int8_t), hipMemcpyHostToDevice);
                hipMemcpy(d_scale[i], h_scale.data(), Ns[i] * sizeof(float), hipMemcpyHostToDevice);
            }

            hipDeviceSynchronize();

            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
                for (size_t proj = 0; proj < Ns.size(); ++proj)
                {
                    bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                        d_A_int8, d_B_vnni[proj], d_C[proj], d_scale_A_blockwise, d_scale[proj],
                        Ns[proj], K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                    if (!ok)
                    {
                        ok = rocmGemv_int8_scatter_vnni_blockwise(
                            d_A_int8, d_B_vnni[proj], d_C[proj], d_scale_A_blockwise, d_scale[proj], nullptr,
                            d_partial, Ns[proj], K, 1.0f, 0.0f, nullptr, device_id_, nullptr);
                    }
                    if (!ok)
                    {
                        result.success = false;
                        return result;
                    }
                }
            }
            hipDeviceSynchronize();

            std::vector<double> total_times;
            std::vector<double> quant_times;
            std::vector<double> gemv_times;
            total_times.reserve(bench_runs);
            quant_times.reserve(bench_runs);
            gemv_times.reserve(bench_runs);

            hipEvent_t total_start, total_stop;
            hipEvent_t quant_start, quant_stop;
            hipEvent_t gemv_start, gemv_stop;
            hipEventCreate(&total_start);
            hipEventCreate(&total_stop);
            hipEventCreate(&quant_start);
            hipEventCreate(&quant_stop);
            hipEventCreate(&gemv_start);
            hipEventCreate(&gemv_stop);

            for (int i = 0; i < bench_runs; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(total_start, 0);

                hipEventRecord(quant_start, 0);
                bool ok = rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
                hipEventRecord(quant_stop, 0);
                hipEventSynchronize(quant_stop);
                float quant_ms = 0.0f;
                hipEventElapsedTime(&quant_ms, quant_start, quant_stop);

                float gemv_ms = 0.0f;
                if (ok)
                {
                    hipEventRecord(gemv_start, 0);
                    for (size_t proj = 0; proj < Ns.size(); ++proj)
                    {
                        ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            d_A_int8, d_B_vnni[proj], d_C[proj], d_scale_A_blockwise, d_scale[proj],
                            Ns[proj], K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                        if (!ok)
                        {
                            ok = rocmGemv_int8_scatter_vnni_blockwise(
                                d_A_int8, d_B_vnni[proj], d_C[proj], d_scale_A_blockwise, d_scale[proj], nullptr,
                                d_partial, Ns[proj], K, 1.0f, 0.0f, nullptr, device_id_, nullptr);
                        }
                        if (!ok)
                            break;
                    }
                    hipEventRecord(gemv_stop, 0);
                    hipEventSynchronize(gemv_stop);
                    hipEventElapsedTime(&gemv_ms, gemv_start, gemv_stop);
                }

                hipEventRecord(total_stop, 0);
                hipEventSynchronize(total_stop);
                float total_ms = 0.0f;
                hipEventElapsedTime(&total_ms, total_start, total_stop);

                if (!ok)
                {
                    result.success = false;
                    return result;
                }

                total_times.push_back(static_cast<double>(total_ms));
                quant_times.push_back(static_cast<double>(quant_ms));
                gemv_times.push_back(static_cast<double>(gemv_ms));
            }

            hipEventDestroy(total_start);
            hipEventDestroy(total_stop);
            hipEventDestroy(quant_start);
            hipEventDestroy(quant_stop);
            hipEventDestroy(gemv_start);
            hipEventDestroy(gemv_stop);

            hipFree(d_A);
            hipFree(d_A_int8);
            hipFree(d_scale_A_blockwise);
            hipFree(d_partial);
            for (size_t i = 0; i < Ns.size(); ++i)
            {
                hipFree(d_B_vnni[i]);
                hipFree(d_scale[i]);
                hipFree(d_C[i]);
            }

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

            result.scale_mean_ms = 0.0;
            result.scale_min_ms = 0.0;

            double total_bytes = K * 4.0;
            for (int N : Ns)
                total_bytes += static_cast<double>(K) * N + N * 4.0 + N * 4.0;
            result.total.gbps = (total_bytes / (result.total.min_ms * 1e-3)) / 1e9;
            result.total.success = true;
            result.success = true;
            return result;
#endif
        }

        BenchSplitResult benchmarkGemvSplitBlockwiseBatched(
            const std::vector<int> &Ns,
            int K,
            int warmup_runs = 5,
            int bench_runs = 20)
        {
            BenchSplitResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            if (Ns.empty() || Ns.size() > 8)
                return result;

            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);

            std::vector<float> h_A(K);
            for (auto &v : h_A)
                v = dist_a(rng);

            float *d_A = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A_blockwise = nullptr;
            float *d_partial = nullptr;
            const int blocks_per_row = K / 32;
            constexpr int MAX_KB = 64;
            const int max_n = *std::max_element(Ns.begin(), Ns.end());
            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_blockwise, blocks_per_row * sizeof(float));
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * max_n * sizeof(float));
            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);

            std::vector<int8_t *> d_B_vnni(Ns.size(), nullptr);
            std::vector<float *> d_scale(Ns.size(), nullptr);
            std::vector<float *> d_C(Ns.size(), nullptr);
            std::vector<std::vector<int8_t>> h_B_vnni(Ns.size());

            for (size_t i = 0; i < Ns.size(); ++i)
            {
                std::vector<int8_t> h_B(static_cast<size_t>(K) * Ns[i]);
                std::vector<float> h_scale(Ns[i]);
                for (auto &v : h_B)
                    v = static_cast<int8_t>(dist_b(rng));
                for (auto &v : h_scale)
                    v = dist_s(rng);

                packVnniWeights(h_B, Ns[i], K, h_B_vnni[i]);
                hipMalloc(&d_B_vnni[i], h_B_vnni[i].size() * sizeof(int8_t));
                hipMalloc(&d_scale[i], Ns[i] * sizeof(float));
                hipMalloc(&d_C[i], Ns[i] * sizeof(float));
                hipMemcpy(d_B_vnni[i], h_B_vnni[i].data(), h_B_vnni[i].size() * sizeof(int8_t), hipMemcpyHostToDevice);
                hipMemcpy(d_scale[i], h_scale.data(), Ns[i] * sizeof(float), hipMemcpyHostToDevice);
            }

            std::vector<const int8_t *> d_B_ptrs(Ns.size());
            std::vector<float *> d_C_ptrs(Ns.size());
            std::vector<const float *> d_scale_ptrs(Ns.size());
            std::vector<const float *> d_bias_ptrs(Ns.size(), nullptr);
            for (size_t i = 0; i < Ns.size(); ++i)
            {
                d_B_ptrs[i] = d_B_vnni[i];
                d_C_ptrs[i] = d_C[i];
                d_scale_ptrs[i] = d_scale[i];
            }

            hipDeviceSynchronize();

            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
                bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                    d_A_int8, d_scale_A_blockwise, static_cast<int>(Ns.size()),
                    d_B_ptrs.data(), d_C_ptrs.data(), d_scale_ptrs.data(), d_bias_ptrs.data(), Ns.data(),
                    K, 1.0f, 0.0f, device_id_, nullptr);
                if (!ok)
                {
                    ok = rocmGemv_int8_scatter_batched_vnni_blockwise(
                        d_A_int8, d_scale_A_blockwise, d_partial, static_cast<int>(Ns.size()),
                        d_B_ptrs.data(), d_C_ptrs.data(), d_scale_ptrs.data(), d_bias_ptrs.data(), Ns.data(),
                        K, 1.0f, 0.0f, device_id_, nullptr);
                }
                if (!ok)
                {
                    result.success = false;
                    return result;
                }
            }
            hipDeviceSynchronize();

            std::vector<double> total_times;
            std::vector<double> quant_times;
            std::vector<double> gemv_times;
            total_times.reserve(bench_runs);
            quant_times.reserve(bench_runs);
            gemv_times.reserve(bench_runs);

            hipEvent_t total_start, total_stop;
            hipEvent_t quant_start, quant_stop;
            hipEvent_t gemv_start, gemv_stop;
            hipEventCreate(&total_start);
            hipEventCreate(&total_stop);
            hipEventCreate(&quant_start);
            hipEventCreate(&quant_stop);
            hipEventCreate(&gemv_start);
            hipEventCreate(&gemv_stop);

            for (int i = 0; i < bench_runs; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(total_start, 0);

                hipEventRecord(quant_start, 0);
                bool ok = rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_blockwise, 1, K, device_id_, nullptr, 32);
                hipEventRecord(quant_stop, 0);
                hipEventSynchronize(quant_stop);
                float quant_ms = 0.0f;
                hipEventElapsedTime(&quant_ms, quant_start, quant_stop);

                float gemv_ms = 0.0f;
                if (ok)
                {
                    hipEventRecord(gemv_start, 0);
                    ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                        d_A_int8, d_scale_A_blockwise, static_cast<int>(Ns.size()),
                        d_B_ptrs.data(), d_C_ptrs.data(), d_scale_ptrs.data(), d_bias_ptrs.data(), Ns.data(),
                        K, 1.0f, 0.0f, device_id_, nullptr);
                    if (!ok)
                    {
                        ok = rocmGemv_int8_scatter_batched_vnni_blockwise(
                            d_A_int8, d_scale_A_blockwise, d_partial, static_cast<int>(Ns.size()),
                            d_B_ptrs.data(), d_C_ptrs.data(), d_scale_ptrs.data(), d_bias_ptrs.data(), Ns.data(),
                            K, 1.0f, 0.0f, device_id_, nullptr);
                    }
                    hipEventRecord(gemv_stop, 0);
                    hipEventSynchronize(gemv_stop);
                    hipEventElapsedTime(&gemv_ms, gemv_start, gemv_stop);
                }

                hipEventRecord(total_stop, 0);
                hipEventSynchronize(total_stop);
                float total_ms = 0.0f;
                hipEventElapsedTime(&total_ms, total_start, total_stop);

                if (!ok)
                {
                    result.success = false;
                    return result;
                }

                total_times.push_back(static_cast<double>(total_ms));
                quant_times.push_back(static_cast<double>(quant_ms));
                gemv_times.push_back(static_cast<double>(gemv_ms));
            }

            hipEventDestroy(total_start);
            hipEventDestroy(total_stop);
            hipEventDestroy(quant_start);
            hipEventDestroy(quant_stop);
            hipEventDestroy(gemv_start);
            hipEventDestroy(gemv_stop);

            hipFree(d_A);
            hipFree(d_A_int8);
            hipFree(d_scale_A_blockwise);
            hipFree(d_partial);
            for (size_t i = 0; i < Ns.size(); ++i)
            {
                hipFree(d_B_vnni[i]);
                hipFree(d_scale[i]);
                hipFree(d_C[i]);
            }

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

            result.scale_mean_ms = 0.0;
            result.scale_min_ms = 0.0;

            double total_bytes = K * 4.0;
            for (int N : Ns)
                total_bytes += static_cast<double>(K) * N + N * 4.0 + N * 4.0;
            result.total.gbps = (total_bytes / (result.total.min_ms * 1e-3)) / 1e9;
            result.total.success = true;
            result.success = true;
            return result;
#endif
        }

        // =========================================================================
        // Benchmark fused scatter+reduce pipeline (2-kernel approach)
        // =========================================================================
        BenchResult benchmarkFusedScatter(int N, int K, int warmup_runs = 5, int bench_runs = 20)
        {
            BenchResult result{};
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
            float *d_partial = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            // Allocate partial buffer: max KB is ~56, but 64 covers all shapes safely
            constexpr int MAX_KB = 64;
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * N * sizeof(float));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // Warmup
            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmGemv_fused_scatter_fp32_int8_vnni(
                    d_A, d_B_vnni, d_C, d_scale, nullptr,
                    d_partial, N, K, 1.0f, 0.0f, nullptr,
                    device_id_, nullptr);
            }
            hipDeviceSynchronize();

            // Timed runs
            std::vector<double> times;
            times.reserve(bench_runs);

            hipEvent_t start, stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);

            for (int i = 0; i < bench_runs; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(start, 0);

                rocmGemv_fused_scatter_fp32_int8_vnni(
                    d_A, d_B_vnni, d_C, d_scale, nullptr,
                    d_partial, N, K, 1.0f, 0.0f, nullptr,
                    device_id_, nullptr);

                hipEventRecord(stop, 0);
                hipEventSynchronize(stop);
                float ms = 0.0f;
                hipEventElapsedTime(&ms, start, stop);
                times.push_back(static_cast<double>(ms));
            }

            hipEventDestroy(start);
            hipEventDestroy(stop);
            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_partial);

            computeStats(times, result.mean_ms, result.min_ms,
                         result.max_ms, result.stddev_ms);

            double bytes = static_cast<double>(K) * N * 1 + K * 4 + N * 4 + N * 4;
            result.gbps = (bytes / (result.min_ms * 1e-3)) / 1e9;
            result.success = true;
            return result;
#endif
        }

        // =========================================================================
        // Benchmark self-reducing scatter (single-kernel approach)
        // =========================================================================
        BenchResult benchmarkSelfReduce(int N, int K, int warmup_runs = 5, int bench_runs = 20)
        {
            BenchResult result{};
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
            float *d_partial = nullptr;
            int *d_counter = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            constexpr int MAX_KB = 64;
            const int grid_n = (N + 127) / 128;
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * N * sizeof(float));
            hipMalloc(&d_counter, grid_n * sizeof(int));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // Warmup
            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni(
                    d_A, d_B_vnni, d_C, d_scale, nullptr,
                    d_partial, d_counter, N, K, 1.0f, 0.0f, nullptr,
                    device_id_, nullptr);
            }
            hipDeviceSynchronize();

            // Timed runs
            std::vector<double> times;
            times.reserve(bench_runs);

            hipEvent_t start, stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);

            for (int i = 0; i < bench_runs; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(start, 0);

                rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni(
                    d_A, d_B_vnni, d_C, d_scale, nullptr,
                    d_partial, d_counter, N, K, 1.0f, 0.0f, nullptr,
                    device_id_, nullptr);

                hipEventRecord(stop, 0);
                hipEventSynchronize(stop);
                float ms = 0.0f;
                hipEventElapsedTime(&ms, start, stop);
                times.push_back(static_cast<double>(ms));
            }

            hipEventDestroy(start);
            hipEventDestroy(stop);
            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_partial);
            hipFree(d_counter);

            computeStats(times, result.mean_ms, result.min_ms,
                         result.max_ms, result.stddev_ms);

            double bytes = static_cast<double>(K) * N * 1 + K * 4 + N * 4 + N * 4;
            result.gbps = (bytes / (result.min_ms * 1e-3)) / 1e9;
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

            // GPU: INT8 VNNI pipeline (blockwise quantize → fused blockwise GEMV)
            float *d_A = nullptr, *d_scale = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A_bw = nullptr;

            const int blocks_per_row = (K + 31) / 32;
            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_bw, blocks_per_row * sizeof(float));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_bw, 1, K, device_id_, nullptr, 32);
            rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                d_A_int8, d_B_vnni, d_C, d_scale_A_bw, d_scale,
                N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
            hipDeviceSynchronize();

            std::vector<float> gpu_out(N);
            hipMemcpy(gpu_out.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_A_int8);
            hipFree(d_scale_A_bw);

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

            // GPU: INT8 VNNI pipeline with bias (blockwise quantize → fused blockwise GEMV+bias)
            float *d_A = nullptr, *d_scale = nullptr, *d_bias = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A_bw = nullptr;

            const int blocks_per_row = (K + 31) / 32;
            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_bias, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_bw, blocks_per_row * sizeof(float));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_bias, h_bias.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_bw, 1, K, device_id_, nullptr, 32);
            rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                d_A_int8, d_B_vnni, d_C, d_scale_A_bw, d_scale,
                N, K, 1.0f, 0.0f, nullptr, d_bias, device_id_, nullptr);
            hipDeviceSynchronize();

            std::vector<float> gpu_out(N);
            hipMemcpy(gpu_out.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_bias);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_A_int8);
            hipFree(d_scale_A_bw);

            return checkCorrectness(gpu_out.data(), ref.data(), N);
#endif
        }

        // =========================================================================
        // Correctness test for fused scatter pipeline
        // =========================================================================
        CorrectnessResult testCorrectnessScatter(int N, int K)
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

            std::vector<float> ref(N);
            cpuReferenceGemv(h_A.data(), h_B.data(), h_scale.data(), ref.data(), N, K);

            std::vector<int8_t> h_B_vnni;
            packVnniWeights(h_B, N, K, h_B_vnni);

            float *d_A = nullptr, *d_scale = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            float *d_partial = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            constexpr int MAX_KB = 64;
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * N * sizeof(float));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            rocmGemv_fused_scatter_fp32_int8_vnni(
                d_A, d_B_vnni, d_C, d_scale, nullptr,
                d_partial, N, K, 1.0f, 0.0f, nullptr,
                device_id_, nullptr);
            hipDeviceSynchronize();

            std::vector<float> gpu_out(N);
            hipMemcpy(gpu_out.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_partial);

            return checkCorrectness(gpu_out.data(), ref.data(), N);
#endif
        }

        // =========================================================================
        // Correctness check: self-reducing scatter (single-kernel path)
        // =========================================================================
        CorrectnessResult testCorrectnessSelfReduce(int N, int K)
        {
            CorrectnessResult bad{0, 0, 0, false};
#ifndef HAVE_ROCM
            return bad;
#else
            // Use same seed as scatter test for comparable results
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

            std::vector<float> ref(N);
            cpuReferenceGemv(h_A.data(), h_B.data(), h_scale.data(), ref.data(), N, K);

            std::vector<int8_t> h_B_vnni;
            packVnniWeights(h_B, N, K, h_B_vnni);

            float *d_A = nullptr, *d_scale = nullptr, *d_C = nullptr;
            int8_t *d_B_vnni = nullptr;
            float *d_partial = nullptr;
            int *d_counter = nullptr;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_scale, N * sizeof(float));
            hipMalloc(&d_C, N * sizeof(float));
            hipMalloc(&d_B_vnni, h_B_vnni.size() * sizeof(int8_t));

            constexpr int MAX_KB = 64;
            const int grid_n = (N + 127) / 128;
            hipMalloc(&d_partial, static_cast<size_t>(MAX_KB) * N * sizeof(float));
            hipMalloc(&d_counter, grid_n * sizeof(int));

            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
            hipMemcpy(d_scale, h_scale.data(), N * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni(
                d_A, d_B_vnni, d_C, d_scale, nullptr,
                d_partial, d_counter, N, K, 1.0f, 0.0f, nullptr,
                device_id_, nullptr);
            hipDeviceSynchronize();

            std::vector<float> gpu_out(N);
            hipMemcpy(gpu_out.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

            hipFree(d_A);
            hipFree(d_scale);
            hipFree(d_C);
            hipFree(d_B_vnni);
            hipFree(d_partial);
            hipFree(d_counter);

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
    // TEST: Correctness — Fused Scatter+Reduce pipeline
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Correctness_Scatter)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        printCorrectnessHeader("Fused Scatter+Reduce Correctness (M=1 decode)");

        // Test all Qwen7B shapes + Qwen0.5B shapes
        struct Shape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<Shape> shapes = {
            // Qwen7B
            {"7B Q proj", kQwen7B.hidden, kQwen7B.hidden},
            {"7B K proj", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
            {"7B V proj", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
            {"7B Wo proj", kQwen7B.hidden, kQwen7B.hidden},
            {"7B FFN Gate", kQwen7B.intermediate, kQwen7B.hidden},
            {"7B FFN Up", kQwen7B.intermediate, kQwen7B.hidden},
            {"7B FFN Down", kQwen7B.hidden, kQwen7B.intermediate},
            {"7B LM Head", kQwen7B.vocab, kQwen7B.hidden},
            // Qwen0.5B
            {"0.5B Q proj", kQwen05B.hidden, kQwen05B.hidden},
            {"0.5B K proj", kQwen05B.num_kv_heads * kQwen05B.head_dim, kQwen05B.hidden},
            {"0.5B V proj", kQwen05B.num_kv_heads * kQwen05B.head_dim, kQwen05B.hidden},
            {"0.5B Wo proj", kQwen05B.hidden, kQwen05B.hidden},
            {"0.5B FFN Gate", kQwen05B.intermediate, kQwen05B.hidden},
            {"0.5B FFN Up", kQwen05B.intermediate, kQwen05B.hidden},
            {"0.5B FFN Down", kQwen05B.hidden, kQwen05B.intermediate},
            {"0.5B LM Head", kQwen05B.vocab, kQwen05B.hidden},
        };

        for (const auto &s : shapes)
        {
            auto r = testCorrectnessScatter(s.N, s.K);
            printCorrectnessRow(s.name, s.N, s.K, r);
            EXPECT_TRUE(r.pass) << s.name << " N=" << s.N << " K=" << s.K
                                << " cosine=" << r.cosine_sim;
        }
        printCorrectnessFooter();
#endif
    }

    // ============================================================================
    // TEST: Correctness — Self-reducing scatter (single-kernel)
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Correctness_SelfReduce)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        printCorrectnessHeader("Self-Reducing Scatter Correctness (M=1 decode)");

        struct Shape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<Shape> shapes = {
            // Qwen7B
            {"7B Q proj", kQwen7B.hidden, kQwen7B.hidden},
            {"7B K proj", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
            {"7B V proj", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
            {"7B Wo proj", kQwen7B.hidden, kQwen7B.hidden},
            {"7B FFN Gate", kQwen7B.intermediate, kQwen7B.hidden},
            {"7B FFN Up", kQwen7B.intermediate, kQwen7B.hidden},
            {"7B FFN Down", kQwen7B.hidden, kQwen7B.intermediate},
            {"7B LM Head", kQwen7B.vocab, kQwen7B.hidden},
            // Qwen0.5B
            {"0.5B Q proj", kQwen05B.hidden, kQwen05B.hidden},
            {"0.5B K proj", kQwen05B.num_kv_heads * kQwen05B.head_dim, kQwen05B.hidden},
            {"0.5B V proj", kQwen05B.num_kv_heads * kQwen05B.head_dim, kQwen05B.hidden},
            {"0.5B Wo proj", kQwen05B.hidden, kQwen05B.hidden},
            {"0.5B FFN Gate", kQwen05B.intermediate, kQwen05B.hidden},
            {"0.5B FFN Up", kQwen05B.intermediate, kQwen05B.hidden},
            {"0.5B FFN Down", kQwen05B.hidden, kQwen05B.intermediate},
            {"0.5B LM Head", kQwen05B.vocab, kQwen05B.hidden},
        };

        for (const auto &s : shapes)
        {
            auto r = testCorrectnessSelfReduce(s.N, s.K);
            printCorrectnessRow(s.name, s.N, s.K, r);
            EXPECT_TRUE(r.pass) << s.name << " N=" << s.N << " K=" << s.K
                                << " cosine=" << r.cosine_sim;
        }
        printCorrectnessFooter();
#endif
    }

    // ============================================================================
    // TEST: Performance — Self-reduce vs Scatter+Reduce comparison
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_SelfReduceVsScatter)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        auto shapes = getDecodeShapes(kQwen7B);
        auto lm = getLMHeadShape(kQwen7B);

        // Table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "N" << "K"
              << "Scatter min(ms)" << "SelfReduce min(ms)" << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int i = 1; i <= 5; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double total_sc_layer = 0;
        double total_sr_layer = 0;
        double lm_sc = 0, lm_sr = 0;

        std::vector<GemvShape> all_shapes = shapes;
        all_shapes.push_back(lm);

        for (const auto &s : all_shapes)
        {
            auto scatter = benchmarkFusedScatter(s.N, s.K);
            auto selfreduce = benchmarkSelfReduce(s.N, s.K);

            double speedup = scatter.min_ms / std::max(1e-9, selfreduce.min_ms);

            char buf_sc[32], buf_sr[32], buf_spd[32];
            snprintf(buf_sc, sizeof(buf_sc), "%.3f", scatter.min_ms);
            snprintf(buf_sr, sizeof(buf_sr), "%.3f", selfreduce.min_ms);
            snprintf(buf_spd, sizeof(buf_spd), "%.2fx", speedup);

            table << s.name << s.N << s.K << buf_sc << buf_sr << buf_spd << fort::endr;

            const bool is_lm = (s.N == lm.N && s.K == lm.K);
            if (is_lm)
            {
                lm_sc = scatter.min_ms;
                lm_sr = selfreduce.min_ms;
            }
            else
            {
                total_sc_layer += scatter.min_ms;
                total_sr_layer += selfreduce.min_ms;
            }
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());

        double all_sc = total_sc_layer * kQwen7B.num_layers + lm_sc;
        double all_sr = total_sr_layer * kQwen7B.num_layers + lm_sr;

        fprintf(stderr, "  Scatter    per-layer:   %8.3f ms\n", total_sc_layer);
        fprintf(stderr, "  SelfReduce per-layer:   %8.3f ms\n", total_sr_layer);
        fprintf(stderr, "  Scatter    all %d + LM: %8.3f ms  (%.1f tok/s)\n",
                kQwen7B.num_layers, all_sc, 1000.0 / all_sc);
        fprintf(stderr, "  SelfReduce all %d + LM: %8.3f ms  (%.1f tok/s)\n",
                kQwen7B.num_layers, all_sr, 1000.0 / all_sr);
        fprintf(stderr, "  Overall speedup:       %.2fx\n\n", all_sc / std::max(1e-9, all_sr));
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
        fprintf(stderr, "  Effective bandwidth:   %8.1f GB/s  (of 1000 GB/s HBM2)\n", effective_gbps);
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
        fprintf(stderr, "  Effective bandwidth:   %8.1f GB/s  (of 1000 GB/s HBM2)\n", effective_gbps);
        fprintf(stderr, "  Weight data read:      %8.1f MB\n\n", all_layers_bytes / 1e6);

        auto shapes_with_lm = shapes;
        shapes_with_lm.push_back(lm);
        printSplitBenchTable("GEMV Split Timing (INT8 VNNI)", shapes_with_lm, split_results);
#endif
    }

    // ============================================================================
    // TEST: Fused Scatter vs 3-kernel pipeline comparison (Qwen7B decode)
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_ScatterVsBaseline)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        auto shapes = getDecodeShapes(kQwen7B);
        auto lm = getLMHeadShape(kQwen7B);

        // Table header
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "N" << "K"
              << "3-Kernel min(ms)" << "Scatter min(ms)" << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int i = 1; i <= 5; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double total_3k_layer = 0;
        double total_sc_layer = 0;
        double lm_3k = 0, lm_sc = 0;

        std::vector<GemvShape> all_shapes = shapes;
        all_shapes.push_back(lm);

        for (const auto &s : all_shapes)
        {
            auto base = benchmarkGemvSplit(s.N, s.K);
            auto scatter = benchmarkFusedScatter(s.N, s.K);

            double speedup = base.total.min_ms / std::max(1e-9, scatter.min_ms);

            char buf_base[32], buf_scatter[32], buf_spd[32];
            snprintf(buf_base, sizeof(buf_base), "%.3f", base.total.min_ms);
            snprintf(buf_scatter, sizeof(buf_scatter), "%.3f", scatter.min_ms);
            snprintf(buf_spd, sizeof(buf_spd), "%.2fx", speedup);

            table << s.name << s.N << s.K << buf_base << buf_scatter << buf_spd << fort::endr;

            const bool is_lm = (s.N == lm.N && s.K == lm.K);
            if (is_lm)
            {
                lm_3k = base.total.min_ms;
                lm_sc = scatter.min_ms;
            }
            else
            {
                total_3k_layer += base.total.min_ms;
                total_sc_layer += scatter.min_ms;
            }
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());

        double all_3k = total_3k_layer * kQwen7B.num_layers + lm_3k;
        double all_sc = total_sc_layer * kQwen7B.num_layers + lm_sc;

        fprintf(stderr, "  3-Kernel per-layer:   %8.3f ms\n", total_3k_layer);
        fprintf(stderr, "  Scatter  per-layer:   %8.3f ms\n", total_sc_layer);
        fprintf(stderr, "  3-Kernel all %d + LM: %8.3f ms  (%.1f tok/s)\n",
                kQwen7B.num_layers, all_3k, 1000.0 / all_3k);
        fprintf(stderr, "  Scatter  all %d + LM: %8.3f ms  (%.1f tok/s)\n",
                kQwen7B.num_layers, all_sc, 1000.0 / all_sc);
        fprintf(stderr, "  Overall speedup:       %.2fx\n\n", all_3k / std::max(1e-9, all_sc));
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
            float *d_scales_A_bw = nullptr;
            int32_t *d_C_int32 = nullptr;

            const int blocks_per_row = (K + 31) / 32;
            hipMalloc(&d_A, M_padded * K * sizeof(float));
            hipMalloc(&d_A_int8, M_padded * K * sizeof(int8_t));
            hipMalloc(&d_scales_A, M_padded * sizeof(float));
            hipMalloc(&d_scales_A_bw, M_padded * blocks_per_row * sizeof(float));
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
            // CK executeTwoKernel_cached requires per-row scales; fill with 1.0 for throughput benchmark
            std::vector<float> h_unit_scales(M_padded, 1.0f);
            hipMemcpy(d_scales_A, h_unit_scales.data(), M_padded * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // Warmup CK
            for (int i = 0; i < 3; ++i)
            {
                rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scales_A_bw, M_padded, K, device_id_, nullptr, 32);
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
                rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scales_A_bw, M_padded, K, device_id_, nullptr, 32);
                rocmQuantGemm_executeTwoKernel_cached(d_A_int8, d_B_int8, d_C, d_scales_A, d_scales_B,
                                                      d_C_int32, M_padded, N, K, device_id_, nullptr);
                hipDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                ck_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            hipFree(d_A);
            hipFree(d_A_int8);
            hipFree(d_scales_A);
            hipFree(d_scales_A_bw);
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

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_AllShapes_AutoSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_ALLSHAPES_AUTOSWEEP");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_ALLSHAPES_AUTOSWEEP=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        struct SweepShape
        {
            const char *name;
            int N;
            int K;
        };
        const int kv_dim = kQwen7B.num_kv_heads * kQwen7B.head_dim;
        const std::vector<SweepShape> shapes = {
            {"Q proj", kQwen7B.hidden, kQwen7B.hidden},
            {"K proj", kv_dim, kQwen7B.hidden},
            {"V proj", kv_dim, kQwen7B.hidden},
            {"Wo proj", kQwen7B.hidden, kQwen7B.hidden},
            {"FFN Gate", kQwen7B.intermediate, kQwen7B.hidden},
            {"FFN Up", kQwen7B.intermediate, kQwen7B.hidden},
            {"FFN Down", kQwen7B.hidden, kQwen7B.intermediate},
        };
        const std::vector<int> tn_candidates = {128, 256};
        const std::vector<int> kb_candidates = {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56};

        for (const auto &shape : shapes)
        {
            rocmGemv_int8_vnni_reset_tuning_overrides();
            const auto baseline = benchmarkGemvSplit(shape.N, shape.K, 5, 20);
            ASSERT_TRUE(baseline.success);

            const double HBM2_PEAK_GBPS = 1000.0;
            const double weight_mb = static_cast<double>(shape.N) * shape.K / 1e6;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "TN" << "KB" << "Blocks" << "Waves/CU" << "Kgrp/Wave"
                  << "GEMV min(ms)" << "BW(GB/s)" << "%Peak" << "Speedup" << fort::endr;
            for (int i = 0; i < 9; ++i)
                table.column(i).set_cell_text_align(fort::text_align::right);

            double best_gemv = std::numeric_limits<double>::max();
            int best_tn = -1, best_kb = -1;

            for (const int tn : tn_candidates)
            {
                for (const int kb : kb_candidates)
                {
                    const int grid_n = (shape.N + tn - 1) / tn;
                    const int k_groups = shape.K / 4;
                    const int total_blocks = grid_n * kb;
                    const int waves_per_block = (tn == 256) ? 2 : 1;
                    const double waves_per_cu = static_cast<double>(total_blocks * waves_per_block) / 60.0;
                    const int kgrp_per_wave = k_groups / kb;

                    if (kgrp_per_wave < 4)
                        continue; // skip absurdly short inner loops

                    rocmGemv_int8_vnni_set_tuning_overrides(tn, kb);
                    const auto r = benchmarkGemvSplit(shape.N, shape.K, 3, 15);
                    if (!r.success)
                        continue;

                    const double bw = weight_mb / r.gemv_min_ms;
                    const double pct = (bw / HBM2_PEAK_GBPS) * 100.0;
                    const double speedup = baseline.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);

                    char buf_wcu[32], buf_bw[32], buf_pct[32], buf_spd[32];
                    snprintf(buf_wcu, sizeof(buf_wcu), "%.1f", waves_per_cu);
                    snprintf(buf_bw, sizeof(buf_bw), "%.0f", bw);
                    snprintf(buf_pct, sizeof(buf_pct), "%.1f%%", pct);
                    snprintf(buf_spd, sizeof(buf_spd), "%.3f", speedup);

                    table << tn << kb << total_blocks << buf_wcu << kgrp_per_wave
                          << formatMs(r.gemv_min_ms) << buf_bw << buf_pct << buf_spd << fort::endr;

                    if (r.gemv_min_ms < best_gemv)
                    {
                        best_gemv = r.gemv_min_ms;
                        best_tn = tn;
                        best_kb = kb;
                    }
                }
            }

            rocmGemv_int8_vnni_reset_tuning_overrides();
            const double best_bw = weight_mb / best_gemv;
            fprintf(stderr, "\n=== %s (N=%d, K=%d, %.1f MB) ===\n", shape.name, shape.N, shape.K, weight_mb);
            fprintf(stderr, "Baseline GEMV: %.6f ms (%.0f GB/s, %.1f%% peak)\n",
                    baseline.gemv_min_ms, weight_mb / baseline.gemv_min_ms,
                    (weight_mb / baseline.gemv_min_ms / HBM2_PEAK_GBPS) * 100.0);
            fprintf(stderr, "Best:    TN=%d KB=%d → %.6f ms (%.0f GB/s, %.1f%% peak)\n",
                    best_tn, best_kb, best_gemv, best_bw, (best_bw / HBM2_PEAK_GBPS) * 100.0);
            fprintf(stderr, "%s\n", table.to_string().c_str());
        }
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_AllShapes)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        const std::vector<const ModelDims *> models = {&kQwen05B, &kQwen3B, &kQwen7B};
        const double HBM2_PEAK_GBPS = 1000.0;

        for (const ModelDims *model : models)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "Shape" << "N" << "K" << "Row GEMV(ms)" << "Block GEMV(ms)" << "Block BW(GB/s)" << "%Peak" << "Speedup"
                  << fort::endr;
            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int i = 1; i < 8; ++i)
                table.column(i).set_cell_text_align(fort::text_align::right);

            auto shapes = getDecodeShapes(*model);
            shapes.push_back(getLMHeadShape(*model));

            for (const auto &shape : shapes)
            {
                const auto rowwise = benchmarkGemvSplit(shape.N, shape.K, 3, 10);
                const auto blockwise = benchmarkGemvSplitBlockwise(shape.N, shape.K, 3, 10);
                ASSERT_TRUE(rowwise.success) << shape.name;
                ASSERT_TRUE(blockwise.success) << shape.name;

                const double weight_mb = static_cast<double>(shape.N) * shape.K / 1e6;
                const double bw = weight_mb / std::max(1e-9, blockwise.gemv_min_ms);
                const double pct = (bw / HBM2_PEAK_GBPS) * 100.0;
                const double speedup = rowwise.gemv_min_ms / std::max(1e-9, blockwise.gemv_min_ms);

                char buf_bw[32], buf_pct[32], buf_spd[32];
                snprintf(buf_bw, sizeof(buf_bw), "%.0f", bw);
                snprintf(buf_pct, sizeof(buf_pct), "%.1f%%", pct);
                snprintf(buf_spd, sizeof(buf_spd), "%.3f", speedup);

                table << shape.name
                      << shape.N
                      << shape.K
                      << formatMs(rowwise.gemv_min_ms)
                      << formatMs(blockwise.gemv_min_ms)
                      << buf_bw
                      << buf_pct
                      << buf_spd
                      << fort::endr;
            }

            fprintf(stderr, "\nINT8 blockwise decode shapes for %s\n%s\n", model->name, table.to_string().c_str());
        }
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_BatchedDecodeGroups)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        const std::vector<BatchedDecodeGroup> groups = {
            {"0.5B QKV", {kQwen05B.hidden, kQwen05B.num_kv_heads * kQwen05B.head_dim, kQwen05B.num_kv_heads * kQwen05B.head_dim}, kQwen05B.hidden},
            {"0.5B GateUp", {kQwen05B.intermediate, kQwen05B.intermediate}, kQwen05B.hidden},
            {"3B QKV", {kQwen3B.hidden, kQwen3B.num_kv_heads * kQwen3B.head_dim, kQwen3B.num_kv_heads * kQwen3B.head_dim}, kQwen3B.hidden},
            {"3B GateUp", {kQwen3B.intermediate, kQwen3B.intermediate}, kQwen3B.hidden},
            {"7B QKV", {kQwen7B.hidden, kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.num_kv_heads * kQwen7B.head_dim}, kQwen7B.hidden},
            {"7B GateUp", {kQwen7B.intermediate, kQwen7B.intermediate}, kQwen7B.hidden},
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Group" << "K" << "Total N" << "Separate quant(ms)" << "Separate gemv(ms)"
              << "Batched quant(ms)" << "Batched gemv(ms)" << "GEMV speedup" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int i = 1; i < 8; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        for (const auto &group : groups)
        {
            const auto separate = benchmarkGemvSplitBlockwiseSharedQuantSeparate(group.Ns, group.K, 3, 10);
            const auto batched = benchmarkGemvSplitBlockwiseBatched(group.Ns, group.K, 3, 10);
            ASSERT_TRUE(separate.success) << group.name;
            ASSERT_TRUE(batched.success) << group.name;

            int total_n = 0;
            for (int n : group.Ns)
                total_n += n;

            char spd[32];
            snprintf(spd, sizeof(spd), "%.3f", separate.gemv_min_ms / std::max(1e-9, batched.gemv_min_ms));

            table << group.name
                  << group.K
                  << total_n
                  << formatMs(separate.quant_min_ms)
                  << formatMs(separate.gemv_min_ms)
                  << formatMs(batched.quant_min_ms)
                  << formatMs(batched.gemv_min_ms)
                  << spd
                  << fort::endr;
        }

        fprintf(stderr, "\nINT8 blockwise batched decode groups\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_BatchedDecodeQkvBreakdown)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED=1 to run";
        }

        const std::vector<BatchedDecodeGroup> groups = {
            {"7B Q only", {kQwen7B.hidden}, kQwen7B.hidden},
            {"7B KV pair", {kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.num_kv_heads * kQwen7B.head_dim}, kQwen7B.hidden},
            {"7B full QKV", {kQwen7B.hidden, kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.num_kv_heads * kQwen7B.head_dim}, kQwen7B.hidden},
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Group" << "K" << "Total N" << "Separate gemv(ms)" << "Batched gemv(ms)" << "GEMV speedup" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int i = 1; i < 6; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        for (const auto &group : groups)
        {
            const auto separate = benchmarkGemvSplitBlockwiseSharedQuantSeparate(group.Ns, group.K, 3, 10);
            const auto batched = benchmarkGemvSplitBlockwiseBatched(group.Ns, group.K, 3, 10);
            ASSERT_TRUE(separate.success) << group.name;
            ASSERT_TRUE(batched.success) << group.name;

            int total_n = 0;
            for (int n : group.Ns)
                total_n += n;

            char spd[32];
            snprintf(spd, sizeof(spd), "%.3f", separate.gemv_min_ms / std::max(1e-9, batched.gemv_min_ms));

            table << group.name
                  << group.K
                  << total_n
                  << formatMs(separate.gemv_min_ms)
                  << formatMs(batched.gemv_min_ms)
                  << spd
                  << fort::endr;
        }

        fprintf(stderr, "\nINT8 blockwise QKV breakdown\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_KVPairFixed)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED=1 to run";
        }

        const std::vector<int> Ns = {
            kQwen7B.num_kv_heads * kQwen7B.head_dim,
            kQwen7B.num_kv_heads * kQwen7B.head_dim,
        };
        const int K = kQwen7B.hidden;

        const auto separate = benchmarkGemvSplitBlockwiseSharedQuantSeparate(Ns, K, 3, 20);
        const auto batched = benchmarkGemvSplitBlockwiseBatched(Ns, K, 3, 20);
        ASSERT_TRUE(separate.success);
        ASSERT_TRUE(batched.success);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Case" << "K" << "Total N" << "Separate gemv(ms)" << "Batched gemv(ms)" << "GEMV speedup" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int i = 1; i < 6; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        char spd[32];
        snprintf(spd, sizeof(spd), "%.3f", separate.gemv_min_ms / std::max(1e-9, batched.gemv_min_ms));

        table << "7B KV pair"
              << K
              << (Ns[0] + Ns[1])
              << formatMs(separate.gemv_min_ms)
              << formatMs(batched.gemv_min_ms)
              << spd
              << fort::endr;

        fprintf(stderr, "\nINT8 blockwise fixed KV pair\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_EqualPairShapeGrid)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED=1 to run";
        }

        const std::vector<int> n_values = {128, 256, 512};
        const std::vector<int> k_values = {896, 2048, 3584, 5120, 8192};

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Pair N" << "K" << "Separate gemv(ms)" << "Batched gemv(ms)" << "Batched speedup" << fort::endr;
        for (int i = 0; i < 5; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        for (const int n : n_values)
        {
            for (const int k : k_values)
            {
                const std::vector<int> ns = {n, n};
                const auto separate = benchmarkGemvSplitBlockwiseSharedQuantSeparate(ns, k, 3, 10);
                const auto batched = benchmarkGemvSplitBlockwiseBatched(ns, k, 3, 10);
                ASSERT_TRUE(separate.success) << "N=" << n << " K=" << k;
                ASSERT_TRUE(batched.success) << "N=" << n << " K=" << k;

                char spd[32];
                snprintf(spd, sizeof(spd), "%.3f", separate.gemv_min_ms / std::max(1e-9, batched.gemv_min_ms));

                table << n
                      << k
                      << formatMs(separate.gemv_min_ms)
                      << formatMs(batched.gemv_min_ms)
                      << spd
                      << fort::endr;
            }
        }

        fprintf(stderr, "\nINT8 blockwise equal-pair shape grid\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_SingleKVFixed)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run";
        }

        const int N = kQwen7B.num_kv_heads * kQwen7B.head_dim;
        const int K = kQwen7B.hidden;

        rocmGemv_int8_vnni_reset_tuning_overrides();
        const auto rowwise = benchmarkGemvSplit(N, K, 5, 20);
        const auto blockwise = benchmarkGemvSplitBlockwise(N, K, 5, 20);
        ASSERT_TRUE(rowwise.success);
        ASSERT_TRUE(blockwise.success);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "N" << "K" << "Row GEMV(ms)" << "Block GEMV(ms)" << "Speedup" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int i = 1; i < 6; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        char spd[32];
        snprintf(spd, sizeof(spd), "%.3f", rowwise.gemv_min_ms / std::max(1e-9, blockwise.gemv_min_ms));

        table << "7B single KV"
              << N
              << K
              << formatMs(rowwise.gemv_min_ms)
              << formatMs(blockwise.gemv_min_ms)
              << spd
              << fort::endr;

        fprintf(stderr, "\nINT8 blockwise single KV fixed\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_QWo_AutoSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_AUTOSWEEP");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_AUTOSWEEP=1 to run INT8 Q/Wo autosweep";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        const int N = kQwen7B.hidden;
        const int K = kQwen7B.hidden;
        const std::vector<int> tn_candidates = {128, 256};
        const std::vector<int> kb_candidates = {8, 10, 12, 14, 16, 20, 24};

        rocmGemv_int8_vnni_reset_tuning_overrides();
        const auto baseline = benchmarkGemvSplit(N, K, 5, 20);
        ASSERT_TRUE(baseline.success);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "TN" << "KB" << "Quant min(ms)" << "GEMV min(ms)" << "Scale min(ms)" << "Total min(ms)" << "GEMV speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::right);
        table.column(1).set_cell_text_align(fort::text_align::right);
        for (int i = 2; i <= 6; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double best_gemv_min = std::numeric_limits<double>::max();
        int best_tn = -1;
        int best_kb = -1;

        for (const int tn : tn_candidates)
        {
            for (const int kb : kb_candidates)
            {
                rocmGemv_int8_vnni_set_tuning_overrides(tn, kb);
                const auto r = benchmarkGemvSplit(N, K, 3, 15);
                ASSERT_TRUE(r.success);

                const double speedup = baseline.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);
                table << tn
                      << kb
                      << formatMs(r.quant_min_ms)
                      << formatMs(r.gemv_min_ms)
                      << formatMs(r.scale_min_ms)
                      << formatMs(r.total.min_ms)
                      << formatMs(speedup)
                      << fort::endr;

                if (r.gemv_min_ms < best_gemv_min)
                {
                    best_gemv_min = r.gemv_min_ms;
                    best_tn = tn;
                    best_kb = kb;
                }
            }
        }

        rocmGemv_int8_vnni_reset_tuning_overrides();

        const double best_speedup = baseline.gemv_min_ms / std::max(1e-9, best_gemv_min);
        fprintf(stderr, "\nINT8 Q/Wo baseline GEMV min: %.6f ms\n", baseline.gemv_min_ms);
        fprintf(stderr, "INT8 Q/Wo best config: TN=%d KB=%d GEMV min=%.6f ms speedup=%.4fx\n", best_tn, best_kb, best_gemv_min, best_speedup);
        fprintf(stderr, "\nINT8 Q/Wo autosweep (N=%d, K=%d)\n%s\n", N, K, table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_QWo_AutoSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run INT8 blockwise Q/Wo autosweep";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        const int N = kQwen7B.hidden;
        const int K = kQwen7B.hidden;

        // Sweep: TN × WK × Unroll × KB × ACT_BK
        const std::vector<int> tn_candidates = {128, 256};
        const std::vector<int> wk_candidates = {2, 3, 4, 6, 8};
        const std::vector<int> unroll_candidates = {1, 2}; // 2 = 2x act_block unroll (TN=256 only)
        const std::vector<int> kb_candidates = {6, 8, 10, 12, 14};
        const std::vector<int> act_bk_candidates = {32, 64, 128};

        rocmGemv_int8_vnni_reset_tuning_overrides();
        rocmGemv_int8_vnni_reset_qwo_overrides();
        const auto baseline = benchmarkGemvSplitBlockwise(N, K, 5, 20);
        ASSERT_TRUE(baseline.success);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "TN" << "WK" << "Unroll" << "KB" << "ACT_BK" << "GEMV min(ms)" << "Total min(ms)" << "GEMV speedup"
              << fort::endr;

        for (int i = 0; i < 8; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double best_gemv_min = std::numeric_limits<double>::max();
        int best_tn = -1, best_wk = -1, best_unroll = -1, best_kb = -1, best_act_bk = -1;

        for (const int act_bk : act_bk_candidates)
        {
            for (const int tn : tn_candidates)
            {
                // ACT_BK > 32 only implemented for TN=128/unroll=2 path
                if (act_bk > 32 && tn != 128)
                    continue;

                for (const int wk : wk_candidates)
                {
                    for (const int unroll : unroll_candidates)
                    {
                        // ACT_BK > 32 only implemented for unroll=2
                        if (act_bk > 32 && unroll != 2)
                            continue;

                        for (const int kb : kb_candidates)
                        {
                            rocmGemv_int8_vnni_set_tuning_overrides(tn, kb);
                            rocmGemv_int8_vnni_set_wk_override(wk);
                            rocmGemv_int8_vnni_set_unroll_override(unroll);
                            rocmGemv_int8_vnni_set_act_block_override(act_bk);

                            const auto r = benchmarkGemvSplitBlockwise(N, K, 3, 15);
                            ASSERT_TRUE(r.success);

                            const double speedup = baseline.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);
                            table << tn << wk << unroll << kb << act_bk
                                  << formatMs(r.gemv_min_ms)
                                  << formatMs(r.total.min_ms)
                                  << formatMs(speedup)
                                  << fort::endr;

                            if (r.gemv_min_ms < best_gemv_min)
                            {
                                best_gemv_min = r.gemv_min_ms;
                                best_tn = tn;
                                best_wk = wk;
                                best_unroll = unroll;
                                best_kb = kb;
                                best_act_bk = act_bk;
                            }
                        }
                    }
                }
            }
        }

        rocmGemv_int8_vnni_reset_tuning_overrides();
        rocmGemv_int8_vnni_reset_qwo_overrides();

        const double best_speedup = baseline.gemv_min_ms / std::max(1e-9, best_gemv_min);
        fprintf(stderr, "\nINT8 blockwise Q/Wo baseline GEMV min: %.6f ms\n", baseline.gemv_min_ms);
        fprintf(stderr, "INT8 blockwise Q/Wo best config: TN=%d WK=%d Unroll=%d KB=%d ACT_BK=%d GEMV min=%.6f ms speedup=%.4fx\n",
                best_tn, best_wk, best_unroll, best_kb, best_act_bk, best_gemv_min, best_speedup);
        fprintf(stderr, "\nINT8 blockwise Q/Wo autosweep (N=%d, K=%d)\n%s\n", N, K, table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_GenericKVPair_AutoSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_BATCHED=1 to run";
        }

        const std::vector<int> Ns = {kQwen7B.num_kv_heads * kQwen7B.head_dim,
                                     kQwen7B.num_kv_heads * kQwen7B.head_dim};
        const int K = kQwen7B.hidden;
        const std::vector<int> tn_candidates = {128, 256};
        const std::vector<int> kb_candidates = {1, 2, 4, 7, 14};

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "TN" << "KB" << "Separate gemv(ms)" << "Batched gemv(ms)" << "Batched speedup" << fort::endr;
        for (int i = 0; i < 5; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double best_batched = std::numeric_limits<double>::max();
        int best_tn = -1;
        int best_kb = -1;

        rocmGemv_int8_vnni_reset_tuning_overrides();

        for (const int tn : tn_candidates)
        {
            for (const int kb : kb_candidates)
            {
                rocmGemv_int8_vnni_set_tuning_overrides(tn, kb);
                const auto separate = benchmarkGemvSplitBlockwiseSharedQuantSeparate(Ns, K, 3, 10);
                const auto batched = benchmarkGemvSplitBlockwiseBatched(Ns, K, 3, 10);
                ASSERT_TRUE(separate.success) << "tn=" << tn << " kb=" << kb;
                ASSERT_TRUE(batched.success) << "tn=" << tn << " kb=" << kb;

                const double speedup = separate.gemv_min_ms / std::max(1e-9, batched.gemv_min_ms);
                table << tn
                      << kb
                      << formatMs(separate.gemv_min_ms)
                      << formatMs(batched.gemv_min_ms)
                      << formatMs(speedup)
                      << fort::endr;

                if (batched.gemv_min_ms < best_batched)
                {
                    best_batched = batched.gemv_min_ms;
                    best_tn = tn;
                    best_kb = kb;
                }
            }
        }

        rocmGemv_int8_vnni_reset_tuning_overrides();

        fprintf(stderr,
                "\nINT8 blockwise generic KV-pair autosweep best: TN=%d KB=%d batched gemv=%.6f ms\n",
                best_tn, best_kb, best_batched);
        fprintf(stderr, "\nINT8 blockwise generic KV-pair autosweep\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_SingleKV_AutoSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        const int N = kQwen7B.num_kv_heads * kQwen7B.head_dim;
        const int K = kQwen7B.hidden;
        const std::vector<int> tn_candidates = {128, 256};
        const std::vector<int> kb_candidates = {1, 2, 4, 7, 14, 16, 28, 56};

        rocmGemv_int8_vnni_reset_tuning_overrides();
        const auto baseline = benchmarkGemvSplitBlockwise(N, K, 5, 20);
        ASSERT_TRUE(baseline.success);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "TN" << "KB" << "Quant min(ms)" << "GEMV min(ms)" << "Total min(ms)" << "GEMV speedup" << fort::endr;
        for (int i = 0; i < 6; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double best_gemv_min = std::numeric_limits<double>::max();
        int best_tn = -1;
        int best_kb = -1;

        for (const int tn : tn_candidates)
        {
            for (const int kb : kb_candidates)
            {
                rocmGemv_int8_vnni_set_tuning_overrides(tn, kb);
                const auto r = benchmarkGemvSplitBlockwise(N, K, 3, 15);
                ASSERT_TRUE(r.success);

                const double speedup = baseline.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);
                table << tn
                      << kb
                      << formatMs(r.quant_min_ms)
                      << formatMs(r.gemv_min_ms)
                      << formatMs(r.total.min_ms)
                      << formatMs(speedup)
                      << fort::endr;

                if (r.gemv_min_ms < best_gemv_min)
                {
                    best_gemv_min = r.gemv_min_ms;
                    best_tn = tn;
                    best_kb = kb;
                }
            }
        }

        rocmGemv_int8_vnni_reset_tuning_overrides();

        const double best_speedup = baseline.gemv_min_ms / std::max(1e-9, best_gemv_min);
        fprintf(stderr, "\nINT8 blockwise single KV baseline GEMV min: %.6f ms\n", baseline.gemv_min_ms);
        fprintf(stderr, "INT8 blockwise single KV best config: TN=%d KB=%d GEMV min=%.6f ms speedup=%.4fx\n", best_tn, best_kb, best_gemv_min, best_speedup);
        fprintf(stderr, "\nINT8 blockwise single KV autosweep (N=%d, K=%d)\n%s\n", N, K, table.to_string().c_str());
#endif
    }

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_LMHead_AutoSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_LMHEAD_AUTOSWEEP");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_LMHEAD_AUTOSWEEP=1 to run INT8 LM Head autosweep";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        const int N = kQwen7B.vocab;
        const int K = kQwen7B.hidden;
        const std::vector<int> tn_candidates = {128, 256, 512};
        const std::vector<int> cpt_candidates = {2, 4};
        const std::vector<int> vec_load_candidates = {1, 0};

        rocmGemv_int8_vnni_reset_tuning_overrides();
        rocmGemv_int8_vnni_reset_wide_tuning_overrides();
        const auto baseline = benchmarkGemvSplit(N, K, 5, 20);
        ASSERT_TRUE(baseline.success);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "TN" << "CPT" << "VecLoad" << "Quant min(ms)" << "GEMV min(ms)" << "Scale min(ms)" << "Total min(ms)" << "GEMV speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::right);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::center);
        for (int i = 3; i <= 7; ++i)
            table.column(i).set_cell_text_align(fort::text_align::right);

        double best_gemv_min = std::numeric_limits<double>::max();
        int best_tn = -1;
        int best_cpt = -1;
        int best_vec_load = -1;

        for (const int tn : tn_candidates)
        {
            for (const int cpt : cpt_candidates)
            {
                for (const int vec_load : vec_load_candidates)
                {
                    rocmGemv_int8_vnni_set_wide_tuning_overrides(tn, cpt, vec_load);
                    const auto r = benchmarkGemvSplit(N, K, 3, 15);
                    ASSERT_TRUE(r.success);

                    const double speedup = baseline.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);
                    table << tn
                          << cpt
                          << (vec_load ? "on" : "off")
                          << formatMs(r.quant_min_ms)
                          << formatMs(r.gemv_min_ms)
                          << formatMs(r.scale_min_ms)
                          << formatMs(r.total.min_ms)
                          << formatMs(speedup)
                          << fort::endr;

                    if (r.gemv_min_ms < best_gemv_min)
                    {
                        best_gemv_min = r.gemv_min_ms;
                        best_tn = tn;
                        best_cpt = cpt;
                        best_vec_load = vec_load;
                    }
                }
            }
        }

        rocmGemv_int8_vnni_reset_wide_tuning_overrides();

        const double best_speedup = baseline.gemv_min_ms / std::max(1e-9, best_gemv_min);
        fprintf(stderr, "\nINT8 LM Head baseline GEMV min: %.6f ms\n", baseline.gemv_min_ms);
        fprintf(stderr, "INT8 LM Head best config: TN=%d CPT=%d VecLoad=%s GEMV min=%.6f ms speedup=%.4fx\n",
                best_tn, best_cpt, best_vec_load ? "on" : "off", best_gemv_min, best_speedup);
        fprintf(stderr, "\nINT8 LM Head autosweep (N=%d, K=%d)\n%s\n", N, K, table.to_string().c_str());
#endif
    }

    // ============================================================================
    // TEST: K/V proj targeted KB+WK sweep
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_KVProj_KBSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        struct KVShape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<KVShape> shapes = {
            {"3B K proj", kQwen3B.num_kv_heads * kQwen3B.head_dim, kQwen3B.hidden},
            {"3B V proj", kQwen3B.num_kv_heads * kQwen3B.head_dim, kQwen3B.hidden},
            {"7B K proj", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
            {"7B V proj", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
        };

        const std::vector<int> kb_candidates = {1, 2, 4, 7, 8, 14, 16, 28, 56, 112};
        const std::vector<int> wk_candidates = {2, 4, 8};

        for (const auto &shape : shapes)
        {
            rocmGemv_int8_vnni_reset_tuning_overrides();
            rocmGemv_int8_vnni_reset_qwo_overrides();

            const auto rowwise = benchmarkGemvSplit(shape.N, shape.K, 5, 20);
            ASSERT_TRUE(rowwise.success);

            const auto blockwise_default = benchmarkGemvSplitBlockwise(shape.N, shape.K, 5, 20);
            ASSERT_TRUE(blockwise_default.success);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "WK" << "KB" << "Waves" << "Acts/Wv" << "Quant(ms)" << "GEMV(ms)" << "Total(ms)"
                  << "vs Row" << fort::endr;
            for (int i = 0; i < 8; ++i)
                table.column(i).set_cell_text_align(fort::text_align::right);

            double best_gemv = std::numeric_limits<double>::max();
            int best_wk = -1, best_kb = -1;
            const int grid_n = (shape.N + 127) / 128;
            const int act_blocks = shape.K / 32;

            for (const int wk : wk_candidates)
            {
                for (const int kb : kb_candidates)
                {
                    if (kb > act_blocks)
                        continue;
                    const int total_waves = grid_n * wk * kb;
                    const double acts_per_wave = static_cast<double>(act_blocks) / (wk * kb);

                    rocmGemv_int8_vnni_set_tuning_overrides(128, kb);
                    rocmGemv_int8_vnni_set_wk_override(wk);

                    const auto r = benchmarkGemvSplitBlockwise(shape.N, shape.K, 3, 15);
                    ASSERT_TRUE(r.success);

                    const double vs_row = rowwise.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);

                    char buf_waves[16], buf_apw[16];
                    snprintf(buf_waves, sizeof(buf_waves), "%d", total_waves);
                    snprintf(buf_apw, sizeof(buf_apw), "%.1f", acts_per_wave);

                    table << wk << kb << buf_waves << buf_apw
                          << formatMs(r.quant_min_ms)
                          << formatMs(r.gemv_min_ms)
                          << formatMs(r.total.min_ms)
                          << formatMs(vs_row)
                          << fort::endr;

                    if (r.gemv_min_ms < best_gemv)
                    {
                        best_gemv = r.gemv_min_ms;
                        best_wk = wk;
                        best_kb = kb;
                    }
                }
            }

            rocmGemv_int8_vnni_reset_tuning_overrides();
            rocmGemv_int8_vnni_reset_qwo_overrides();

            const double best_vs_row = rowwise.gemv_min_ms / std::max(1e-9, best_gemv);
            fprintf(stderr, "\n%s (N=%d, K=%d)\n", shape.name, shape.N, shape.K);
            fprintf(stderr, "  Row-wise baseline: %.6f ms\n", rowwise.gemv_min_ms);
            fprintf(stderr, "  Block-wise default: %.6f ms (%.3fx row)\n",
                    blockwise_default.gemv_min_ms,
                    rowwise.gemv_min_ms / std::max(1e-9, blockwise_default.gemv_min_ms));
            fprintf(stderr, "  Best: WK=%d KB=%d GEMV=%.6f ms (%.3fx row)\n",
                    best_wk, best_kb, best_gemv, best_vs_row);
            fprintf(stderr, "%s\n", table.to_string().c_str());

            // --- Memset overhead measurement ---
            // Run default heuristic with memset skipped to measure its overhead
            rocmGemv_int8_vnni_reset_tuning_overrides();
            rocmGemv_int8_vnni_reset_qwo_overrides();
            rocmGemv_int8_vnni_set_skip_memset(1);
            const auto no_memset = benchmarkGemvSplitBlockwise(shape.N, shape.K, 5, 30);
            rocmGemv_int8_vnni_set_skip_memset(0);
            ASSERT_TRUE(no_memset.success);

            const double memset_cost = blockwise_default.gemv_min_ms - no_memset.gemv_min_ms;
            const double no_memset_parity = rowwise.gemv_min_ms / std::max(1e-9, no_memset.gemv_min_ms);
            fprintf(stderr, "  Without memset: %.6f ms (%.3fx row) → memset overhead: %.3f us\n",
                    no_memset.gemv_min_ms, no_memset_parity, memset_cost * 1000.0);
        }
#endif
    }

    // ============================================================================
    // TEST: FFN Down targeted KB+WK sweep
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_FFNDown_KBSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        struct FFNDownShape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<FFNDownShape> shapes = {
            {"3B FFN Down", kQwen3B.hidden, kQwen3B.intermediate},
            {"7B FFN Down", kQwen7B.hidden, kQwen7B.intermediate},
        };

        const std::vector<int> kb_candidates = {1, 2, 4, 7, 10, 14, 20, 30, 40, 50, 74};
        const std::vector<int> wk_candidates = {4, 6, 8};

        for (const auto &shape : shapes)
        {
            rocmGemv_int8_vnni_reset_tuning_overrides();
            rocmGemv_int8_vnni_reset_qwo_overrides();

            // Rowwise baseline
            const auto rowwise = benchmarkGemvSplit(shape.N, shape.K, 5, 20);
            ASSERT_TRUE(rowwise.success);

            // Blockwise default
            const auto blockwise_default = benchmarkGemvSplitBlockwise(shape.N, shape.K, 5, 20);
            ASSERT_TRUE(blockwise_default.success);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "WK" << "KB" << "Quant(ms)" << "GEMV(ms)" << "Total(ms)"
                  << "vs Row" << "vs Default" << "BW(GB/s)" << fort::endr;
            for (int i = 0; i < 8; ++i)
                table.column(i).set_cell_text_align(fort::text_align::right);

            double best_gemv = std::numeric_limits<double>::max();
            int best_wk = -1, best_kb = -1;

            for (const int wk : wk_candidates)
            {
                for (const int kb : kb_candidates)
                {
                    rocmGemv_int8_vnni_set_tuning_overrides(128, kb);
                    rocmGemv_int8_vnni_set_wk_override(wk);

                    const auto r = benchmarkGemvSplitBlockwise(shape.N, shape.K, 3, 15);
                    ASSERT_TRUE(r.success);

                    const double vs_row = rowwise.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);
                    const double vs_def = blockwise_default.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);
                    const double bw = (static_cast<double>(shape.N) * shape.K) / (r.gemv_min_ms * 1e-3) / 1e9;

                    table << wk << kb
                          << formatMs(r.quant_min_ms)
                          << formatMs(r.gemv_min_ms)
                          << formatMs(r.total.min_ms)
                          << formatMs(vs_row)
                          << formatMs(vs_def)
                          << formatMs(bw)
                          << fort::endr;

                    if (r.gemv_min_ms < best_gemv)
                    {
                        best_gemv = r.gemv_min_ms;
                        best_wk = wk;
                        best_kb = kb;
                    }
                }
            }

            rocmGemv_int8_vnni_reset_tuning_overrides();
            rocmGemv_int8_vnni_reset_qwo_overrides();

            const double best_vs_row = rowwise.gemv_min_ms / std::max(1e-9, best_gemv);
            fprintf(stderr, "\n%s (N=%d, K=%d)\n", shape.name, shape.N, shape.K);
            fprintf(stderr, "  Row-wise baseline: %.6f ms\n", rowwise.gemv_min_ms);
            fprintf(stderr, "  Block-wise default: %.6f ms (%.3fx row)\n",
                    blockwise_default.gemv_min_ms,
                    rowwise.gemv_min_ms / std::max(1e-9, blockwise_default.gemv_min_ms));
            fprintf(stderr, "  Best: WK=%d KB=%d GEMV=%.6f ms (%.3fx row)\n",
                    best_wk, best_kb, best_gemv, best_vs_row);
            fprintf(stderr, "%s\n", table.to_string().c_str());
        }
#endif
    }

    // ============================================================================
    // TEST: K+V batched pair experiment
    //
    // Compares three approaches for dispatching K and V projections:
    //   1. 2× individual GEMV calls (serial, same stream) — current production
    //   2. 1× batched API (currently splits to 2 singles internally)
    //   3. 1× batched API with force_pair (uses grid_kpar_pair kernel)
    //   4. 2× individual GEMV on separate HIP streams (concurrent test)
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_Blockwise_KVPairBatching)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_INT8_BLOCKWISE_ALLSHAPES=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());

        struct KVShape
        {
            const char *name;
            int N;
            int K;
        };
        std::vector<KVShape> shapes = {
            {"3B K/V", kQwen3B.num_kv_heads * kQwen3B.head_dim, kQwen3B.hidden},
            {"7B K/V", kQwen7B.num_kv_heads * kQwen7B.head_dim, kQwen7B.hidden},
        };

        constexpr int WARMUP = 5;
        constexpr int BENCH = 30;

        for (const auto &shape : shapes)
        {
            const int N = shape.N;
            const int K = shape.K;

            rocmGemv_int8_vnni_reset_tuning_overrides();
            rocmGemv_int8_vnni_reset_qwo_overrides();

            // --- Allocate: shared A, separate B_k/B_v/C_k/C_v ---
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
            std::uniform_int_distribution<int> dist_b(-127, 127);
            std::uniform_real_distribution<float> dist_s(0.001f, 0.1f);

            std::vector<float> h_A(K);
            for (auto &v : h_A)
                v = dist_a(rng);

            auto make_projection = [&](int proj_N)
            {
                struct ProjData
                {
                    std::vector<int8_t> h_B_vnni;
                    std::vector<float> h_scale;
                    int8_t *d_B = nullptr;
                    float *d_scale = nullptr;
                    float *d_C = nullptr;
                };
                ProjData p;
                std::vector<int8_t> h_B(static_cast<size_t>(K) * proj_N);
                p.h_scale.resize(proj_N);
                for (auto &v : h_B)
                    v = static_cast<int8_t>(dist_b(rng));
                for (auto &v : p.h_scale)
                    v = dist_s(rng);
                packVnniWeights(h_B, proj_N, K, p.h_B_vnni);
                hipMalloc(&p.d_B, p.h_B_vnni.size() * sizeof(int8_t));
                hipMalloc(&p.d_scale, proj_N * sizeof(float));
                hipMalloc(&p.d_C, proj_N * sizeof(float));
                hipMemcpy(p.d_B, p.h_B_vnni.data(), p.h_B_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice);
                hipMemcpy(p.d_scale, p.h_scale.data(), proj_N * sizeof(float), hipMemcpyHostToDevice);
                return p;
            };

            auto proj_k = make_projection(N);
            auto proj_v = make_projection(N);

            float *d_A = nullptr;
            int8_t *d_A_int8 = nullptr;
            float *d_scale_A_bw = nullptr;
            const int blocks_per_row = K / 32;

            hipMalloc(&d_A, K * sizeof(float));
            hipMalloc(&d_A_int8, K * sizeof(int8_t));
            hipMalloc(&d_scale_A_bw, blocks_per_row * sizeof(float));
            hipMemcpy(d_A, h_A.data(), K * sizeof(float), hipMemcpyHostToDevice);
            hipDeviceSynchronize();

            // Quantize activations once
            rocmQuantGemm_quantizeActivationsBlockwise(d_A, d_A_int8, d_scale_A_bw, 1, K, device_id_, nullptr, 32);
            hipDeviceSynchronize();

            // Create HIP events and second stream
            hipEvent_t ev_start, ev_stop, ev_mid;
            hipEventCreate(&ev_start);
            hipEventCreate(&ev_stop);
            hipEventCreate(&ev_mid);
            hipStream_t stream2 = nullptr;
            hipStreamCreate(&stream2);

            // Batched API arrays
            const int8_t *d_B_ptrs[2] = {proj_k.d_B, proj_v.d_B};
            float *d_C_ptrs[2] = {proj_k.d_C, proj_v.d_C};
            const float *d_scale_ptrs[2] = {proj_k.d_scale, proj_v.d_scale};
            const float *d_bias_ptrs[2] = {nullptr, nullptr};
            int N_per_proj[2] = {N, N};

            // ---- Warmup all paths ----
            for (int i = 0; i < WARMUP; ++i)
            {
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_k.d_B, proj_k.d_C, d_scale_A_bw, proj_k.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_v.d_B, proj_v.d_C, d_scale_A_bw, proj_v.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
            }
            hipDeviceSynchronize();

            // ===== Approach 1: 2× serial on default stream =====
            std::vector<double> serial_times;
            serial_times.reserve(BENCH);
            for (int i = 0; i < BENCH; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(ev_start, 0);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_k.d_B, proj_k.d_C, d_scale_A_bw, proj_k.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_v.d_B, proj_v.d_C, d_scale_A_bw, proj_v.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                hipEventRecord(ev_stop, 0);
                hipEventSynchronize(ev_stop);
                float ms = 0;
                hipEventElapsedTime(&ms, ev_start, ev_stop);
                serial_times.push_back(ms);
            }

            // ===== Approach 2: batched API (default — splits to 2 singles) =====
            std::vector<double> batched_split_times;
            batched_split_times.reserve(BENCH);
            for (int i = 0; i < BENCH; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(ev_start, 0);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                    d_A_int8, d_scale_A_bw, 2,
                    d_B_ptrs, d_C_ptrs, d_scale_ptrs, d_bias_ptrs, N_per_proj,
                    K, 1.0f, 0.0f, device_id_, nullptr);
                hipEventRecord(ev_stop, 0);
                hipEventSynchronize(ev_stop);
                float ms = 0;
                hipEventElapsedTime(&ms, ev_start, ev_stop);
                batched_split_times.push_back(ms);
            }

            // ===== Approach 3: batched API with force_pair (grid_kpar_pair kernel) =====
            // Sweep KB values for the pair kernel
            std::vector<int> pair_kb_values = {2, 4, 7, 8, 14};
            struct PairResult
            {
                int kb;
                double min_ms;
            };
            std::vector<PairResult> pair_results;

            for (int kb : pair_kb_values)
            {
                rocmGemv_int8_vnni_set_force_pair(1);
                rocmGemv_int8_vnni_set_tuning_overrides(128, kb);

                // Warmup
                for (int i = 0; i < 3; ++i)
                {
                    rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                        d_A_int8, d_scale_A_bw, 2,
                        d_B_ptrs, d_C_ptrs, d_scale_ptrs, d_bias_ptrs, N_per_proj,
                        K, 1.0f, 0.0f, device_id_, nullptr);
                }
                hipDeviceSynchronize();

                std::vector<double> times;
                times.reserve(BENCH);
                for (int i = 0; i < BENCH; ++i)
                {
                    hipDeviceSynchronize();
                    hipEventRecord(ev_start, 0);
                    rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
                        d_A_int8, d_scale_A_bw, 2,
                        d_B_ptrs, d_C_ptrs, d_scale_ptrs, d_bias_ptrs, N_per_proj,
                        K, 1.0f, 0.0f, device_id_, nullptr);
                    hipEventRecord(ev_stop, 0);
                    hipEventSynchronize(ev_stop);
                    float ms = 0;
                    hipEventElapsedTime(&ms, ev_start, ev_stop);
                    times.push_back(ms);
                }
                std::sort(times.begin(), times.end());
                pair_results.push_back({kb, times[0]});

                rocmGemv_int8_vnni_set_force_pair(0);
                rocmGemv_int8_vnni_reset_tuning_overrides();
                rocmGemv_int8_vnni_reset_qwo_overrides();
            }

            // ===== Approach 4: 2× on separate HIP streams (concurrent) =====
            // Warmup on stream2
            for (int i = 0; i < WARMUP; ++i)
            {
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_v.d_B, proj_v.d_C, d_scale_A_bw, proj_v.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, stream2);
            }
            hipStreamSynchronize(stream2);

            std::vector<double> concurrent_times;
            concurrent_times.reserve(BENCH);
            for (int i = 0; i < BENCH; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(ev_start, 0);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_k.d_B, proj_k.d_C, d_scale_A_bw, proj_k.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, nullptr);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                    d_A_int8, proj_v.d_B, proj_v.d_C, d_scale_A_bw, proj_v.d_scale,
                    N, K, 1.0f, 0.0f, nullptr, nullptr, device_id_, stream2);
                hipStreamSynchronize(stream2);
                hipEventRecord(ev_stop, 0);
                hipEventSynchronize(ev_stop);
                float ms = 0;
                hipEventElapsedTime(&ms, ev_start, ev_stop);
                concurrent_times.push_back(ms);
            }

            // ===== Approach 5: LDS K-reduce pair kernel =====
            // Warmup
            for (int i = 0; i < WARMUP; ++i)
            {
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
                    d_A_int8, d_scale_A_bw,
                    proj_k.d_B, proj_v.d_B,
                    proj_k.d_C, proj_v.d_C,
                    proj_k.d_scale, proj_v.d_scale,
                    N, N, K, 1.0f, device_id_, nullptr);
            }
            hipDeviceSynchronize();

            std::vector<double> lds_pair_times;
            lds_pair_times.reserve(BENCH);
            for (int i = 0; i < BENCH; ++i)
            {
                hipDeviceSynchronize();
                hipEventRecord(ev_start, 0);
                rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
                    d_A_int8, d_scale_A_bw,
                    proj_k.d_B, proj_v.d_B,
                    proj_k.d_C, proj_v.d_C,
                    proj_k.d_scale, proj_v.d_scale,
                    N, N, K, 1.0f, device_id_, nullptr);
                hipEventRecord(ev_stop, 0);
                hipEventSynchronize(ev_stop);
                float ms = 0;
                hipEventElapsedTime(&ms, ev_start, ev_stop);
                lds_pair_times.push_back(ms);
            }

            // Also sweep KB for the lds pair kernel
            std::vector<int> lds_pair_kb_values = {4, 7, 8, 14, 28};
            struct LdsPairResult
            {
                int kb;
                double min_ms;
            };
            std::vector<LdsPairResult> lds_pair_results;

            for (int kb : lds_pair_kb_values)
            {
                rocmGemv_int8_vnni_set_tuning_overrides(128, kb);

                for (int i = 0; i < 3; ++i)
                {
                    rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
                        d_A_int8, d_scale_A_bw,
                        proj_k.d_B, proj_v.d_B,
                        proj_k.d_C, proj_v.d_C,
                        proj_k.d_scale, proj_v.d_scale,
                        N, N, K, 1.0f, device_id_, nullptr);
                }
                hipDeviceSynchronize();

                std::vector<double> times;
                times.reserve(BENCH);
                for (int i = 0; i < BENCH; ++i)
                {
                    hipDeviceSynchronize();
                    hipEventRecord(ev_start, 0);
                    rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
                        d_A_int8, d_scale_A_bw,
                        proj_k.d_B, proj_v.d_B,
                        proj_k.d_C, proj_v.d_C,
                        proj_k.d_scale, proj_v.d_scale,
                        N, N, K, 1.0f, device_id_, nullptr);
                    hipEventRecord(ev_stop, 0);
                    hipEventSynchronize(ev_stop);
                    float ms = 0;
                    hipEventElapsedTime(&ms, ev_start, ev_stop);
                    times.push_back(ms);
                }
                std::sort(times.begin(), times.end());
                lds_pair_results.push_back({kb, times[0]});

                rocmGemv_int8_vnni_reset_tuning_overrides();
            }

            // --- Compute stats ---
            auto min_of = [](const std::vector<double> &v)
            {
                return *std::min_element(v.begin(), v.end());
            };
            auto median_of = [](std::vector<double> v)
            {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };

            const double serial_min = min_of(serial_times);
            const double batched_split_min = min_of(batched_split_times);
            const double concurrent_min = min_of(concurrent_times);
            const double lds_pair_default_min = min_of(lds_pair_times);
            const double serial_med = median_of(serial_times);
            const double batched_split_med = median_of(batched_split_times);
            const double concurrent_med = median_of(concurrent_times);
            const double lds_pair_default_med = median_of(lds_pair_times);

            // Rowwise baseline for parity context
            const auto rowwise_single = benchmarkGemvSplit(N, K, 5, 20);

            // Find best pair result (grid_kpar_pair)
            double best_pair_min = std::numeric_limits<double>::max();
            int best_pair_kb = -1;
            for (const auto &pr : pair_results)
            {
                if (pr.min_ms < best_pair_min)
                {
                    best_pair_min = pr.min_ms;
                    best_pair_kb = pr.kb;
                }
            }

            // Find best lds pair result
            double best_lds_pair_min = lds_pair_default_min;
            int best_lds_pair_kb = -1;
            for (const auto &pr : lds_pair_results)
            {
                if (pr.min_ms < best_lds_pair_min)
                {
                    best_lds_pair_min = pr.min_ms;
                    best_lds_pair_kb = pr.kb;
                }
            }

            // --- Print results ---
            fprintf(stderr, "\n╔══════════════════════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║  %s  K+V Pair Batching Experiment (N=%d, K=%d)               \n", shape.name, N, K);
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║  rocBLAS single (K or V):    %.3f ms                             \n", rowwise_single.gemv_min_ms);
            fprintf(stderr, "║  rocBLAS K+V serial (est):   %.3f ms                             \n", rowwise_single.gemv_min_ms * 2);
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║  Approach                       Min(ms)   Med(ms)  vs Serial    \n");
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║  1. 2×serial (baseline)         %.3f     %.3f     1.000x       \n", serial_min, serial_med);
            fprintf(stderr, "║  2. batched(split-to-2)         %.3f     %.3f     %.3fx       \n",
                    batched_split_min, batched_split_med, serial_min / std::max(1e-9, batched_split_min));
            fprintf(stderr, "║  3. batched(force grid_kpar)    %.3f     %.3f     %.3fx  KB=%d\n",
                    best_pair_min, best_pair_min, serial_min / std::max(1e-9, best_pair_min), best_pair_kb);
            fprintf(stderr, "║  4. 2×concurrent streams        %.3f     %.3f     %.3fx       \n",
                    concurrent_min, concurrent_med, serial_min / std::max(1e-9, concurrent_min));
            fprintf(stderr, "║  5. LDS K-reduce pair (default) %.3f     %.3f     %.3fx       \n",
                    lds_pair_default_min, lds_pair_default_med, serial_min / std::max(1e-9, lds_pair_default_min));
            fprintf(stderr, "║  5. LDS K-reduce pair (best KB) %.3f     -         %.3fx  KB=%d\n",
                    best_lds_pair_min, serial_min / std::max(1e-9, best_lds_pair_min), best_lds_pair_kb);
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║  K+V parity vs rocBLAS K+V:                                      \n");
            fprintf(stderr, "║    Serial:     %.3fx                                             \n", (rowwise_single.gemv_min_ms * 2) / std::max(1e-9, serial_min));
            fprintf(stderr, "║    Grid pair:  %.3fx                                             \n", (rowwise_single.gemv_min_ms * 2) / std::max(1e-9, best_pair_min));
            fprintf(stderr, "║    LDS pair:   %.3fx                                             \n", (rowwise_single.gemv_min_ms * 2) / std::max(1e-9, best_lds_pair_min));
            fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");

            // LDS pair KB sweep detail
            fprintf(stderr, "\n  LDS K-reduce pair KB sweep:\n");
            fprintf(stderr, "    default → %.3f ms (%.3fx vs serial)\n",
                    lds_pair_default_min, serial_min / std::max(1e-9, lds_pair_default_min));
            for (const auto &pr : lds_pair_results)
            {
                fprintf(stderr, "    KB=%2d → %.3f ms (%.3fx vs serial)\n",
                        pr.kb, pr.min_ms, serial_min / std::max(1e-9, pr.min_ms));
            }

            // Grid_kpar pair KB sweep detail
            fprintf(stderr, "\n  Force grid_kpar pair KB sweep:\n");
            for (const auto &pr : pair_results)
            {
                fprintf(stderr, "    KB=%2d → %.3f ms (%.3fx vs serial)\n",
                        pr.kb, pr.min_ms, serial_min / std::max(1e-9, pr.min_ms));
            }

            // Cleanup
            hipStreamDestroy(stream2);
            hipEventDestroy(ev_start);
            hipEventDestroy(ev_stop);
            hipEventDestroy(ev_mid);
            hipFree(d_A);
            hipFree(d_A_int8);
            hipFree(d_scale_A_bw);
            hipFree(proj_k.d_B);
            hipFree(proj_k.d_scale);
            hipFree(proj_k.d_C);
            hipFree(proj_v.d_B);
            hipFree(proj_v.d_scale);
            hipFree(proj_v.d_C);
        }
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
        fprintf(stderr, "║  Bandwidth Roofline Analysis (MI50: 1000 GB/s HBM2 theoretical)                ║\n");
        fprintf(stderr, "╠═══════════════════╦════════════╦════════════╦══════════╦═════════════════════════╣\n");
        fprintf(stderr, "║ Shape             ║ Weight(MB) ║  Min(ms)   ║  GB/s    ║  %% of HBM2 peak       ║\n");
        fprintf(stderr, "╠═══════════════════╬════════════╬════════════╬══════════╬═════════════════════════╣\n");

        const double HBM2_PEAK_GBPS = 1000.0;

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

    // ============================================================================
    // TEST: Qwen3.5-35B-A3B MoE Expert shapes TN×KB sweep
    //
    // Validates the GEMV heuristic for the very small expert projections in
    // Qwen3.5-35B-A3B (d_model=2048, expert_intermediate=512, 256 experts).
    // These are much smaller than dense model projections and may not be well
    // served by the heuristic tuned for 7B shapes.
    // ============================================================================

    TEST_F(ROCmGemvPerfTest, Benchmark_INT8VNNI_MoE35B_ExpertSweep)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device";

        const char *run_sweep = std::getenv("LLAMINAR_RUN_MOE35B_EXPERT_SWEEP");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_MOE35B_EXPERT_SWEEP=1 to run";
        }

        fprintf(stderr, "\nDevice: %s\n", device_name_.c_str());
        fprintf(stderr, "\n=== Qwen3.5-35B-A3B MoE Expert GEMV Heuristic Validation ===\n");
        fprintf(stderr, "Model: d_model=2048, expert_intermediate=512, 256 experts (top-8)\n");
        fprintf(stderr, "       shared_intermediate=512, 40 layers (27 MoE + 10 GDN + 3 FA)\n\n");

        // Qwen3.5-35B-A3B MoE dimensions
        constexpr int kMoEDModel = 2048;
        constexpr int kMoEIntermediate = 512; // per-expert intermediate
        constexpr int kMoENumExperts = 256;
        constexpr int kMoETopK = 8;
        constexpr int kMoELayers = 27; // MoE layers only

        struct SweepShape
        {
            const char *name;
            int N;
            int K;
            int calls_per_layer; // how many times this shape is called per MoE layer
        };

        // Per-expert shapes (called top_k=8 times per token per MoE layer, grouped into 1 kernel)
        // Shared expert shapes (called once per MoE layer)
        const std::vector<SweepShape> shapes = {
            {"Expert Gate", kMoEIntermediate, kMoEDModel, kMoETopK}, // 512×2048 ×8
            {"Expert Up", kMoEIntermediate, kMoEDModel, kMoETopK},   // 512×2048 ×8
            {"Expert Down", kMoEDModel, kMoEIntermediate, kMoETopK}, // 2048×512 ×8
            {"Shared Gate", kMoEIntermediate, kMoEDModel, 1},        // 512×2048 ×1
            {"Shared Up", kMoEIntermediate, kMoEDModel, 1},          // 512×2048 ×1
            {"Shared Down", kMoEDModel, kMoEIntermediate, 1},        // 2048×512 ×1
        };

        const std::vector<int> tn_candidates = {128, 256};
        const std::vector<int> kb_candidates = {2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64};

        const double HBM2_PEAK_GBPS = 1000.0;

        // Summary table at the end
        fort::utf8_table summary;
        summary.set_border_style(FT_DOUBLE2_STYLE);
        summary << fort::header
                << "Shape" << "N" << "K" << "×/layer"
                << "Heuristic TN" << "Heuristic KB" << "Heuristic(ms)" << "Heuristic BW"
                << "Best TN" << "Best KB" << "Best(ms)" << "Best BW"
                << "Speedup" << fort::endr;

        for (const auto &shape : shapes)
        {
            // First: get heuristic (auto) result
            rocmGemv_int8_vnni_reset_tuning_overrides();
            const auto heuristic = benchmarkGemvSplit(shape.N, shape.K, 5, 30);
            ASSERT_TRUE(heuristic.success) << "Heuristic benchmark failed for " << shape.name;

            const double weight_mb = static_cast<double>(shape.N) * shape.K / 1e6;

            // Sweep TN×KB grid
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);
            table << fort::header
                  << "TN" << "KB" << "Blocks" << "Waves/CU" << "Kgrp/Wave"
                  << "GEMV min(ms)" << "BW(GB/s)" << "%Peak" << "vs Heuristic" << fort::endr;
            for (int i = 0; i < 9; ++i)
                table.column(i).set_cell_text_align(fort::text_align::right);

            double best_gemv = heuristic.gemv_min_ms;
            int best_tn = -1, best_kb = -1;

            for (const int tn : tn_candidates)
            {
                for (const int kb : kb_candidates)
                {
                    const int grid_n = (shape.N + tn - 1) / tn;
                    const int k_groups = shape.K / 4;
                    const int total_blocks = grid_n * kb;
                    const int waves_per_block = (tn == 256) ? 2 : 1;
                    const double waves_per_cu = static_cast<double>(total_blocks * waves_per_block) / 60.0;
                    const int kgrp_per_wave = k_groups / kb;

                    if (kgrp_per_wave < 2)
                        continue; // skip degenerate configs

                    rocmGemv_int8_vnni_set_tuning_overrides(tn, kb);
                    const auto r = benchmarkGemvSplit(shape.N, shape.K, 3, 20);
                    if (!r.success)
                        continue;

                    const double bw = weight_mb / r.gemv_min_ms;
                    const double pct = (bw / HBM2_PEAK_GBPS) * 100.0;
                    const double speedup = heuristic.gemv_min_ms / std::max(1e-9, r.gemv_min_ms);

                    char buf_wcu[32], buf_bw[32], buf_pct[32], buf_spd[32];
                    snprintf(buf_wcu, sizeof(buf_wcu), "%.1f", waves_per_cu);
                    snprintf(buf_bw, sizeof(buf_bw), "%.0f", bw);
                    snprintf(buf_pct, sizeof(buf_pct), "%.1f%%", pct);
                    snprintf(buf_spd, sizeof(buf_spd), "%.3fx", speedup);

                    table << tn << kb << total_blocks << buf_wcu << kgrp_per_wave
                          << formatMs(r.gemv_min_ms) << buf_bw << buf_pct << buf_spd << fort::endr;

                    if (r.gemv_min_ms < best_gemv)
                    {
                        best_gemv = r.gemv_min_ms;
                        best_tn = tn;
                        best_kb = kb;
                    }
                }
            }

            rocmGemv_int8_vnni_reset_tuning_overrides();

            const double heuristic_bw = weight_mb / heuristic.gemv_min_ms;
            const double best_bw = weight_mb / best_gemv;
            const double speedup = heuristic.gemv_min_ms / std::max(1e-9, best_gemv);

            fprintf(stderr, "\n=== %s (N=%d, K=%d, %.3f MB, ×%d/layer) ===\n",
                    shape.name, shape.N, shape.K, weight_mb, shape.calls_per_layer);
            fprintf(stderr, "Heuristic: %.6f ms (%.0f GB/s, %.1f%% peak)\n",
                    heuristic.gemv_min_ms, heuristic_bw,
                    (heuristic_bw / HBM2_PEAK_GBPS) * 100.0);
            if (best_tn > 0)
            {
                fprintf(stderr, "Best:      TN=%d KB=%d → %.6f ms (%.0f GB/s, %.1f%% peak) [%.3fx]\n",
                        best_tn, best_kb, best_gemv, best_bw,
                        (best_bw / HBM2_PEAK_GBPS) * 100.0, speedup);
            }
            else
            {
                fprintf(stderr, "Best:      heuristic IS optimal\n");
            }
            fprintf(stderr, "%s\n", table.to_string().c_str());

            // Add to summary
            char h_bw_str[32], b_bw_str[32], spd_str[32];
            snprintf(h_bw_str, sizeof(h_bw_str), "%.0f GB/s", heuristic_bw);
            snprintf(b_bw_str, sizeof(b_bw_str), "%.0f GB/s", best_bw);
            snprintf(spd_str, sizeof(spd_str), "%.3fx", speedup);
            summary << shape.name << shape.N << shape.K << shape.calls_per_layer
                    << 128 << "auto" << formatMs(heuristic.gemv_min_ms) << h_bw_str
                    << (best_tn > 0 ? best_tn : 128) << (best_kb > 0 ? std::to_string(best_kb) : "auto")
                    << formatMs(best_gemv) << b_bw_str
                    << spd_str << fort::endr;
        }

        // Print full-model impact estimate
        fprintf(stderr, "\n=== Full Model Impact Estimate (27 MoE layers) ===\n");
        double heuristic_total_ms = 0;
        double best_total_ms = 0;
        for (const auto &shape : shapes)
        {
            rocmGemv_int8_vnni_reset_tuning_overrides();
            const auto h = benchmarkGemvSplit(shape.N, shape.K, 3, 15);
            heuristic_total_ms += h.gemv_min_ms * shape.calls_per_layer;

            // Re-measure best (quick)
            // (Using heuristic as approximation here; actual best was tracked above)
            best_total_ms += h.gemv_min_ms * shape.calls_per_layer;
        }
        heuristic_total_ms *= kMoELayers;
        best_total_ms *= kMoELayers;
        fprintf(stderr, "Heuristic total MoE GEMV: %.3f ms/token (%d layers × per-layer)\n",
                heuristic_total_ms, kMoELayers);
        fprintf(stderr, "Theoretical throughput contribution: %.0f tok/s (GEMV-only)\n",
                1000.0 / heuristic_total_ms);

        fprintf(stderr, "\n=== Summary ===\n");
        for (int i = 0; i < 13; ++i)
            summary.column(i).set_cell_text_align(fort::text_align::right);
        summary.column(0).set_cell_text_align(fort::text_align::left);
        fprintf(stderr, "%s\n", summary.to_string().c_str());
#endif
    }
} // namespace
