/**
 * @file Perf__ROCmFP16vsINT8Gemm.cpp
 * @brief Performance comparison: INT8 CK vs FP16 hipBLAS GEMM on ROCm
 *
 * This benchmark tests the hypothesis that FP16 GEMM via hipBLAS may be faster
 * than INT8 GEMM via ComposableKernel on gfx906 (MI50/MI60) GPUs.
 *
 * The test compares:
 *   1. INT8 Two-Kernel (CK GEMM + scale kernel) - Current default
 *   2. INT8 hipBLAS fallback
 *   3. FP16 hipBLAS (new path being benchmarked)
 *
 * For each path, we measure:
 *   - End-to-end time (including data conversion)
 *   - Kernel-only time (GEMM execution)
 *   - Throughput (TFLOPS)
 *   - Accuracy (cosine similarity vs FP32 reference)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_map>

#include "kernels/rocm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"
#include "utils/Logger.h"

#ifdef HAVE_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class ROCmFP16vsINT8Perf : public ::testing::Test
{
protected:
    bool has_rocm_device_ = false;
    std::string device_name_;
    std::string arch_name_;

    void SetUp() override
    {
#ifdef HAVE_ROCM
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        has_rocm_device_ = (err == hipSuccess && device_count > 0);

        if (has_rocm_device_)
        {
            hipDeviceProp_t props;
            (void)hipGetDeviceProperties(&props, 0);
            device_name_ = props.name;
            arch_name_ = props.gcnArchName;
        }
#endif
    }

    /**
     * @brief Compute cosine similarity
     */
    double cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
    }

    /**
     * @brief Compute FP32 reference GEMM using OneDNN for raw INT8 matmul
     *
     * C[m,n] = sum_k(A[m,k] * B[k,n]) * scaleA[m] * scaleB[n]
     *
     * Uses OneDNN's optimized INT8 matmul for the core GEMM, then applies
     * scales manually in a separate pass.
     */
    void computeReference(
        const std::vector<int8_t> &A, // [M × K]
        const std::vector<int8_t> &B, // [K × N]
        const std::vector<float> &scaleA,
        const std::vector<float> &scaleB,
        std::vector<float> &C,
        int M, int N, int K)
    {
        C.resize(static_cast<size_t>(M) * N);

#ifdef HAVE_ONEDNN
        using namespace dnnl;
        using dt = memory::data_type;
        using tag = memory::format_tag;

        // Thread-local engine and stream
        static thread_local engine eng(engine::kind::cpu, 0);
        static thread_local stream strm(eng);

        // Memory descriptors: A[M,K], B[K,N], C[M,N]
        memory::dims src_dims = {M, K};
        memory::dims weights_dims = {K, N};
        memory::dims dst_dims = {M, N};

        auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
        auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab);
        auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

        // Create matmul primitive descriptor (no scales - raw INT8 GEMM)
        auto matmul_pd = matmul::primitive_desc(eng, src_md, weights_md, dst_md);

        // Create memory objects
        auto src_mem = memory(src_md, eng, const_cast<int8_t *>(A.data()));
        auto weights_mem = memory(weights_md, eng, const_cast<int8_t *>(B.data()));
        auto dst_mem = memory(dst_md, eng, C.data());

        // Execute: C = A @ B (raw INT8 dot products -> FP32)
        auto matmul_prim = matmul(matmul_pd);
        matmul_prim.execute(strm, {{DNNL_ARG_SRC, src_mem},
                                   {DNNL_ARG_WEIGHTS, weights_mem},
                                   { DNNL_ARG_DST,
                                     dst_mem }});
        strm.wait();

// Apply scales manually: C[m,n] *= scaleA[m] * scaleB[n]
#pragma omp parallel for collapse(2) schedule(static)
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                C[static_cast<size_t>(m) * N + n] *= scaleA[m] * scaleB[n];
            }
        }
#else
// Fallback: OpenMP parallelized naive implementation
#pragma omp parallel for collapse(2) schedule(static)
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                int32_t acc = 0;
                for (int k = 0; k < K; ++k)
                {
                    acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
                           static_cast<int32_t>(B[static_cast<size_t>(k) * N + n]);
                }
                C[static_cast<size_t>(m) * N + n] =
                    static_cast<float>(acc) * scaleA[m] * scaleB[n];
            }
        }
#endif
    }

    /**
     * @brief Generate random INT8 data and scales
     */
    void generateTestData(
        std::vector<int8_t> &A,
        std::vector<int8_t> &B,
        std::vector<float> &scaleA,
        std::vector<float> &scaleB,
        int M, int N, int K)
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);
        std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

        A.resize(static_cast<size_t>(M) * K);
        B.resize(static_cast<size_t>(K) * N);
        scaleA.resize(M);
        scaleB.resize(N);

        for (auto &v : A)
            v = static_cast<int8_t>(int8_dist(rng));
        for (auto &v : B)
            v = static_cast<int8_t>(int8_dist(rng));
        for (auto &v : scaleA)
            v = scale_dist(rng);
        for (auto &v : scaleB)
            v = scale_dist(rng);
    }

    /**
     * @brief Print benchmark header
     */
    void printHeader(const std::string &title)
    {
        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
                  << "║  " << std::left << std::setw(122) << title << "║\n"
                  << "║  Device: " << std::left << std::setw(114) << (device_name_ + " (" + arch_name_ + ")") << "║\n"
                  << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
                  << "║  " << std::setw(18) << "Path"
                  << " │ " << std::setw(20) << "Dimensions"
                  << " │ " << std::setw(14) << "Total (ms)"
                  << " │ " << std::setw(14) << "GEMM (ms)"
                  << " │ " << std::setw(12) << "TFLOPS"
                  << " │ " << std::setw(10) << "Cosine"
                  << " │ " << std::setw(10) << "Speedup"
                  << "║\n"
                  << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
                  << std::endl;
    }

    void printFooter()
    {
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"
                  << std::endl;
    }

    /**
     * @brief Print result row
     */
    void printRow(
        const std::string &path,
        int M, int N, int K,
        double total_ms,
        double gemm_ms,
        double tflops,
        double cosine,
        double speedup = 1.0)
    {
        std::ostringstream dims;
        dims << "M=" << M << " N=" << N << " K=" << K;

        std::cout << "║  " << std::left << std::setw(18) << path
                  << " │ " << std::setw(20) << dims.str()
                  << " │ " << std::fixed << std::setprecision(3) << std::setw(14) << total_ms
                  << " │ " << std::setw(14) << gemm_ms
                  << " │ " << std::setprecision(3) << std::setw(12) << tflops
                  << " │ " << std::setprecision(6) << std::setw(10) << cosine
                  << " │ " << std::setprecision(2) << std::setw(10) << (speedup == 1.0 ? "-" : std::to_string(speedup) + "x")
                  << "║" << std::endl;
    }

    /**
     * @brief Benchmark INT8 Two-Kernel path
     */
    double benchmarkINT8TwoKernel(
        const std::vector<int8_t> &h_A,
        const std::vector<int8_t> &h_B,
        const std::vector<float> &h_scaleA,
        const std::vector<float> &h_scaleB,
        std::vector<float> &h_output,
        int M, int N, int K,
        int warmup, int iters,
        double &cosine,
        const std::vector<float> &reference)
    {
#ifdef HAVE_ROCM
        // Allocate device memory
        int8_t *d_A = nullptr, *d_B = nullptr;
        float *d_scaleA = nullptr, *d_scaleB = nullptr, *d_E = nullptr;

        hipMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(int8_t));
        hipMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(int8_t));
        hipMalloc(&d_scaleA, static_cast<size_t>(M) * sizeof(float));
        hipMalloc(&d_scaleB, static_cast<size_t>(N) * sizeof(float));
        hipMalloc(&d_E, static_cast<size_t>(M) * N * sizeof(float));

        hipMemcpy(d_A, h_A.data(), static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), static_cast<size_t>(K) * N * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_scaleA, h_scaleA.data(), static_cast<size_t>(M) * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_scaleB, h_scaleB.data(), static_cast<size_t>(N) * sizeof(float), hipMemcpyHostToDevice);

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            rocmQuantGemm_executeTwoKernel(d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, 0);
        }
        hipDeviceSynchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            rocmQuantGemm_executeTwoKernel(d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, 0);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

        // Get output for accuracy check
        h_output.resize(static_cast<size_t>(M) * N);
        hipMemcpy(h_output.data(), d_E, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToHost);
        cosine = cosineSimilarity(h_output, reference);

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_scaleA);
        hipFree(d_scaleB);
        hipFree(d_E);

        return total_ms;
#else
        return 0.0;
#endif
    }

    /**
     * @brief Benchmark INT8 hipBLAS path
     */
    double benchmarkINT8HipBLAS(
        const std::vector<int8_t> &h_A,
        const std::vector<int8_t> &h_B,
        const std::vector<float> &h_scaleA,
        const std::vector<float> &h_scaleB,
        std::vector<float> &h_output,
        int M, int N, int K,
        int warmup, int iters,
        double &cosine,
        const std::vector<float> &reference)
    {
#ifdef HAVE_ROCM
        // Allocate device memory
        int8_t *d_A = nullptr, *d_B = nullptr;
        float *d_scaleA = nullptr, *d_scaleB = nullptr, *d_E = nullptr;

        hipMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(int8_t));
        hipMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(int8_t));
        hipMalloc(&d_scaleA, static_cast<size_t>(M) * sizeof(float));
        hipMalloc(&d_scaleB, static_cast<size_t>(N) * sizeof(float));
        hipMalloc(&d_E, static_cast<size_t>(M) * N * sizeof(float));

        hipMemcpy(d_A, h_A.data(), static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), static_cast<size_t>(K) * N * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_scaleA, h_scaleA.data(), static_cast<size_t>(M) * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_scaleB, h_scaleB.data(), static_cast<size_t>(N) * sizeof(float), hipMemcpyHostToDevice);

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            rocmQuantGemm_executeHipBLAS(d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, 0);
        }
        hipDeviceSynchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            rocmQuantGemm_executeHipBLAS(d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, 0);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

        // Get output
        h_output.resize(static_cast<size_t>(M) * N);
        hipMemcpy(h_output.data(), d_E, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToHost);
        cosine = cosineSimilarity(h_output, reference);

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_scaleA);
        hipFree(d_scaleB);
        hipFree(d_E);

        return total_ms;
#else
        return 0.0;
#endif
    }

    /**
     * @brief Benchmark FP16 path
     */
    double benchmarkFP16(
        const std::vector<int8_t> &h_A,
        const std::vector<int8_t> &h_B,
        const std::vector<float> &h_scaleA,
        const std::vector<float> &h_scaleB,
        std::vector<float> &h_output,
        int M, int N, int K,
        int warmup, int iters,
        double &cosine,
        const std::vector<float> &reference)
    {
#ifdef HAVE_ROCM
        // Allocate device memory
        int8_t *d_A = nullptr, *d_B = nullptr;
        float *d_scaleA = nullptr, *d_scaleB = nullptr, *d_E = nullptr;

        hipMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(int8_t));
        hipMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(int8_t));
        hipMalloc(&d_scaleA, static_cast<size_t>(M) * sizeof(float));
        hipMalloc(&d_scaleB, static_cast<size_t>(N) * sizeof(float));
        hipMalloc(&d_E, static_cast<size_t>(M) * N * sizeof(float));

        hipMemcpy(d_A, h_A.data(), static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), static_cast<size_t>(K) * N * sizeof(int8_t), hipMemcpyHostToDevice);
        hipMemcpy(d_scaleA, h_scaleA.data(), static_cast<size_t>(M) * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_scaleB, h_scaleB.data(), static_cast<size_t>(N) * sizeof(float), hipMemcpyHostToDevice);

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            rocmQuantGemm_executeFP16(d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, 0, nullptr);
        }
        hipDeviceSynchronize();

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            rocmQuantGemm_executeFP16(d_A, d_B, d_E, d_scaleA, d_scaleB, M, N, K, 0, nullptr);
        }
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

        // Get output
        h_output.resize(static_cast<size_t>(M) * N);
        hipMemcpy(h_output.data(), d_E, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToHost);
        cosine = cosineSimilarity(h_output, reference);

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_scaleA);
        hipFree(d_scaleB);
        hipFree(d_E);

        return total_ms;
#else
        return 0.0;
#endif
    }

    /**
     * @brief Run comparison for a single configuration
     */
    void runComparison(int M, int N, int K, int warmup = 5, int iters = 20)
    {
        std::vector<int8_t> h_A, h_B;
        std::vector<float> h_scaleA, h_scaleB;
        generateTestData(h_A, h_B, h_scaleA, h_scaleB, M, N, K);

        // Compute reference
        std::vector<float> reference;
        computeReference(h_A, h_B, h_scaleA, h_scaleB, reference, M, N, K);

        std::vector<float> output;
        double cosine;
        double ops = 2.0 * M * N * K;

        // INT8 Two-Kernel (baseline)
        double int8_2k_ms = benchmarkINT8TwoKernel(h_A, h_B, h_scaleA, h_scaleB, output, M, N, K, warmup, iters, cosine, reference);
        double int8_2k_tflops = (ops / (int8_2k_ms * 1e-3)) / 1e12;
        printRow("INT8 Two-Kernel", M, N, K, int8_2k_ms, int8_2k_ms, int8_2k_tflops, cosine, 1.0);

        // INT8 hipBLAS
        double int8_hb_ms = benchmarkINT8HipBLAS(h_A, h_B, h_scaleA, h_scaleB, output, M, N, K, warmup, iters, cosine, reference);
        double int8_hb_tflops = (ops / (int8_hb_ms * 1e-3)) / 1e12;
        double int8_hb_speedup = int8_2k_ms / int8_hb_ms;
        printRow("INT8 hipBLAS", M, N, K, int8_hb_ms, int8_hb_ms, int8_hb_tflops, cosine, int8_hb_speedup);

        // FP16 hipBLAS
        double fp16_ms = benchmarkFP16(h_A, h_B, h_scaleA, h_scaleB, output, M, N, K, warmup, iters, cosine, reference);
        double fp16_tflops = (ops / (fp16_ms * 1e-3)) / 1e12;
        double fp16_speedup = int8_2k_ms / fp16_ms;
        printRow("FP16 hipBLAS", M, N, K, fp16_ms, fp16_ms, fp16_tflops, cosine, fp16_speedup);

        std::cout << "╠──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╣" << std::endl;
    }
};

// ============================================================================
// Tests
// ============================================================================

/**
 * @test Compare INT8 vs FP16 on Qwen2.5-0.5B dimensions
 *
 * Hidden=896, Intermediate=4864
 */
TEST_F(ROCmFP16vsINT8Perf, Qwen0_5B_Comparison)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";

    printHeader("FP16 vs INT8 Comparison: Qwen2.5-0.5B (hidden=896, intermediate=4864)");

    // Prefill batch sizes
    runComparison(128, 896, 896);  // Attn output M=128
    runComparison(128, 4864, 896); // FFN Up M=128
    runComparison(128, 896, 4864); // FFN Down M=128

    // Large prefill
    runComparison(512, 896, 896);
    runComparison(512, 4864, 896);
    runComparison(512, 896, 4864);

    printFooter();
}

/**
 * @test Compare INT8 vs FP16 on Qwen2.5-7B dimensions
 *
 * Hidden=3584, Intermediate=18944
 */
TEST_F(ROCmFP16vsINT8Perf, Qwen7B_Comparison)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";

    printHeader("FP16 vs INT8 Comparison: Qwen2.5-7B (hidden=3584, intermediate=18944)");

    // Moderate prefill
    runComparison(128, 3584, 3584);  // Attn output
    runComparison(128, 18944, 3584); // FFN Up
    runComparison(128, 3584, 18944); // FFN Down

    // Large prefill
    runComparison(512, 3584, 3584);
    runComparison(512, 18944, 3584);
    runComparison(512, 3584, 18944);

    printFooter();
}

/**
 * @test Compare INT8 vs FP16 across batch sizes
 *
 * Tests scaling behavior as M increases
 */
TEST_F(ROCmFP16vsINT8Perf, BatchSizeScaling)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";

    printHeader("FP16 vs INT8: Batch Size Scaling (FFN Up, Qwen-7B dimensions)");

    // N=18944, K=3584 (FFN Up projection)
    for (int M : {64, 128, 256, 512, 1024})
    {
        runComparison(M, 18944, 3584);
    }

    printFooter();
}

/**
 * @test Quick smoke test for FP16 path
 */
TEST_F(ROCmFP16vsINT8Perf, FP16_SmokeTest)
{
    if (!has_rocm_device_)
        GTEST_SKIP() << "No ROCm device";

    const int M = 128, N = 256, K = 128;

    std::vector<int8_t> h_A, h_B;
    std::vector<float> h_scaleA, h_scaleB;
    generateTestData(h_A, h_B, h_scaleA, h_scaleB, M, N, K);

    // Reference
    std::vector<float> reference;
    computeReference(h_A, h_B, h_scaleA, h_scaleB, reference, M, N, K);

    // Run FP16 path
    std::vector<float> output;
    double cosine;
    double fp16_ms = benchmarkFP16(h_A, h_B, h_scaleA, h_scaleB, output, M, N, K, 1, 1, cosine, reference);

    // Debug: check for NaN/Inf and print sample values
    int nan_count = 0, inf_count = 0;
    for (const auto &v : output)
    {
        if (std::isnan(v))
            nan_count++;
        if (std::isinf(v))
            inf_count++;
    }
    LOG_INFO("[FP16 Smoke Test] Time: " << fp16_ms << " ms, Cosine: " << cosine);
    LOG_INFO("[FP16 Smoke Test] NaN count: " << nan_count << ", Inf count: " << inf_count);
    LOG_INFO("[FP16 Smoke Test] Reference[0:5]: " << reference[0] << ", " << reference[1] << ", "
                                                  << reference[2] << ", " << reference[3] << ", " << reference[4]);
    LOG_INFO("[FP16 Smoke Test] Output[0:5]: " << output[0] << ", " << output[1] << ", "
                                               << output[2] << ", " << output[3] << ", " << output[4]);
    LOG_INFO("[FP16 Smoke Test] Scale A[0:3]: " << h_scaleA[0] << ", " << h_scaleA[1] << ", " << h_scaleA[2]);
    LOG_INFO("[FP16 Smoke Test] Scale B[0:3]: " << h_scaleB[0] << ", " << h_scaleB[1] << ", " << h_scaleB[2]);

    // FP16 should have good accuracy (close to FP32 reference)
    EXPECT_GT(cosine, 0.999) << "FP16 accuracy too low";
}
