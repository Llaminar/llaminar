/**
 * @file Test__JitWoProjectionOptimized.cpp
 * @brief Unit tests for optimized Wo projection BLAS-style microkernels
 *
 * Tests the JitWoProjectionOptimizedEmitter which provides:
 *   - emit_gemv_4x64_fp32: 4×64 GEMV for decode (M=1)
 *   - emit_gemm_microkernel_6x16: 6×16 microkernel for prefill
 *
 * These optimized kernels target 5-6× speedup over naive row-by-row dot product.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/attention/q8_1/jit/JitMicrokernelBase.h"
#include "kernels/cpu/attention/q8_1/jit/JitWoProjectionOptimized.h"

#include <cmath>
#include <random>
#include <vector>
#include <cstring>
#include <chrono>
#include <iomanip>

using namespace llaminar::v2::kernels::jit;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__JitWoProjectionOptimized : public ::testing::Test
{
protected:
    // Reference GEMV: output[N] = context[K] × Wo[K,N]
    void reference_gemv(const float *context, const float *Wo, float *output,
                        int K, int N)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += context[k] * Wo[k * N + n];
            }
            output[n] = sum;
        }
    }

    // Reference GEMM: C[M,N] = A[M,K] × B[K,N]
    void reference_gemm(const float *A, const float *B, float *C,
                        int M, int K, int N)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = sum;
            }
        }
    }

    float compute_max_abs_error(const float *a, const float *b, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float err = std::fabs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    float compute_rms_error(const float *a, const float *b, size_t n)
    {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / n);
    }

    // Reference GEMV for fused Wo semantics: output[rows] = Wo[rows, cols] × context[cols]
    void reference_gemv_wox(const float *context, const float *Wo_rowmajor, float *output,
                            int rows, int cols)
    {
        for (int i = 0; i < rows; ++i)
        {
            float sum = 0.0f;
            const float *row = Wo_rowmajor + i * cols;
            for (int k = 0; k < cols; ++k)
            {
                sum += row[k] * context[k];
            }
            output[i] = sum;
        }
    }
};

// ============================================================================
// GEMV 4x64 Tests (Decode Path)
// ============================================================================

TEST_F(Test__JitWoProjectionOptimized, GEMV_SmallMatrix_64x64)
{
    constexpr int K = 64;
    constexpr int N = 64;

    // Allocate aligned buffers
    std::vector<float> context(K);
    std::vector<float> Wo(K * N);
    std::vector<float> output_jit(N, 0.0f);
    std::vector<float> output_ref(N, 0.0f);

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < K; ++i)
        context[i] = dist(rng);
    for (int i = 0; i < K * N; ++i)
        Wo[i] = dist(rng);

    // Compute reference
    reference_gemv(context.data(), Wo.data(), output_ref.data(), K, N);

    // Create JIT kernel
    class GemvKernel : public JitMicrokernelBase
    {
    public:
        GemvKernel(int K, int N) : JitMicrokernelBase(), K_(K), N_(N)
        {
            generate_code();
        }

        void call(const float *context, const float *Wo, float *output)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(context, Wo, output);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_4x64_fp32(*this, rdi, rsi, rdx, K_, N_);
            ret();
            ready();
        }
        int K_, N_;
    };

    GemvKernel kernel(K, N);
    kernel.call(context.data(), Wo.data(), output_jit.data());

    // Verify correctness
    float max_err = compute_max_abs_error(output_jit.data(), output_ref.data(), N);
    float rms_err = compute_rms_error(output_jit.data(), output_ref.data(), N);

    EXPECT_LT(max_err, 1e-4f) << "Max absolute error too large for 64x64 GEMV";
    EXPECT_LT(rms_err, 1e-5f) << "RMS error too large for 64x64 GEMV";
}

TEST_F(Test__JitWoProjectionOptimized, GEMV_QwenSize_896x896)
{
    // Qwen 0.5B has d_model=896
    constexpr int K = 896;
    constexpr int N = 896;

    std::vector<float> context(K);
    std::vector<float> Wo(K * N);
    std::vector<float> output_jit(N, 0.0f);
    std::vector<float> output_ref(N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < K; ++i)
        context[i] = dist(rng);
    for (int i = 0; i < K * N; ++i)
        Wo[i] = dist(rng);

    reference_gemv(context.data(), Wo.data(), output_ref.data(), K, N);

    class GemvKernel : public JitMicrokernelBase
    {
    public:
        GemvKernel(int K, int N) : K_(K), N_(N)
        {
            generate_code();
        }
        void call(const float *context, const float *Wo, float *output)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(context, Wo, output);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_4x64_fp32(*this, rdi, rsi, rdx, K_, N_);
            ret();
            ready();
        }
        int K_, N_;
    };

    GemvKernel kernel(K, N);
    kernel.call(context.data(), Wo.data(), output_jit.data());

    float max_err = compute_max_abs_error(output_jit.data(), output_ref.data(), N);
    float rms_err = compute_rms_error(output_jit.data(), output_ref.data(), N);

    // With K=896, accumulation errors grow
    EXPECT_LT(max_err, 1e-3f) << "Max error too large for 896x896 GEMV";
    EXPECT_LT(rms_err, 1e-4f) << "RMS error too large for 896x896 GEMV";
}

TEST_F(Test__JitWoProjectionOptimized, GEMV_NonMultiple64_800x900)
{
    // Test non-aligned dimensions
    constexpr int K = 800;
    constexpr int N = 900;

    std::vector<float> context(K);
    std::vector<float> Wo(K * N);
    std::vector<float> output_jit(N, 0.0f);
    std::vector<float> output_ref(N, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < K; ++i)
        context[i] = dist(rng);
    for (int i = 0; i < K * N; ++i)
        Wo[i] = dist(rng);

    reference_gemv(context.data(), Wo.data(), output_ref.data(), K, N);

    class GemvKernel : public JitMicrokernelBase
    {
    public:
        GemvKernel(int K, int N) : K_(K), N_(N)
        {
            generate_code();
        }
        void call(const float *context, const float *Wo, float *output)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(context, Wo, output);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_4x64_fp32(*this, rdi, rsi, rdx, K_, N_);
            ret();
            ready();
        }
        int K_, N_;
    };

    GemvKernel kernel(K, N);
    kernel.call(context.data(), Wo.data(), output_jit.data());

    float max_err = compute_max_abs_error(output_jit.data(), output_ref.data(), N);
    EXPECT_LT(max_err, 1e-3f) << "Max error too large for non-aligned GEMV";
}

// ============================================================================
// GEMV Wo@x Row-Major Tests (Matches fused Wo semantics)
// ============================================================================

TEST_F(Test__JitWoProjectionOptimized, GEMV_WoX_RowMajor_64x64)
{
    constexpr int rows = 64;
    constexpr int cols = 64;

    std::vector<float> context(cols);
    std::vector<float> Wo(rows * cols);
    std::vector<float> output_jit(rows, 0.0f);
    std::vector<float> output_ref(rows, 0.0f);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < cols; ++i)
        context[i] = dist(rng);
    for (int i = 0; i < rows * cols; ++i)
        Wo[i] = dist(rng);

    reference_gemv_wox(context.data(), Wo.data(), output_ref.data(), rows, cols);

    class GemvWoXKernel : public JitMicrokernelBase
    {
    public:
        GemvWoXKernel(int rows, int cols) : rows_(rows), cols_(cols)
        {
            generate_code();
        }
        void call(const float *context, const float *Wo, float *output)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(context, Wo, output);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_wox_rowmajor_fp32(*this, rdi, rsi, rdx, rows_, cols_);
            ret();
            ready();
        }
        int rows_, cols_;
    };

    GemvWoXKernel kernel(rows, cols);
    kernel.call(context.data(), Wo.data(), output_jit.data());

    float max_err = compute_max_abs_error(output_jit.data(), output_ref.data(), rows);
    float rms_err = compute_rms_error(output_jit.data(), output_ref.data(), rows);

    EXPECT_LT(max_err, 1e-4f) << "Max error too large for Wo@x 64x64";
    EXPECT_LT(rms_err, 1e-5f) << "RMS error too large for Wo@x 64x64";
}

TEST_F(Test__JitWoProjectionOptimized, GEMV_WoX_RowMajor_Tails_31x47)
{
    // Exercise both row remainder and col tail paths
    constexpr int rows = 31;
    constexpr int cols = 47;

    std::vector<float> context(cols);
    std::vector<float> Wo(rows * cols);
    std::vector<float> output_jit(rows, 0.0f);
    std::vector<float> output_ref(rows, 0.0f);

    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < cols; ++i)
        context[i] = dist(rng);
    for (int i = 0; i < rows * cols; ++i)
        Wo[i] = dist(rng);

    reference_gemv_wox(context.data(), Wo.data(), output_ref.data(), rows, cols);

    class GemvWoXKernel : public JitMicrokernelBase
    {
    public:
        GemvWoXKernel(int rows, int cols) : rows_(rows), cols_(cols)
        {
            generate_code();
        }
        void call(const float *context, const float *Wo, float *output)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(context, Wo, output);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_wox_rowmajor_fp32(*this, rdi, rsi, rdx, rows_, cols_);
            ret();
            ready();
        }
        int rows_, cols_;
    };

    GemvWoXKernel kernel(rows, cols);
    kernel.call(context.data(), Wo.data(), output_jit.data());

    float max_err = compute_max_abs_error(output_jit.data(), output_ref.data(), rows);
    EXPECT_LT(max_err, 1e-3f) << "Max error too large for Wo@x tails";
}

// ============================================================================
// GEMM 6x16 Microkernel Tests (Prefill Path)
// ============================================================================

TEST_F(Test__JitWoProjectionOptimized, GEMM_Microkernel_6x16)
{
    constexpr int M = 6;  // Microkernel tile M
    constexpr int N = 16; // Microkernel tile N
    constexpr int K = 64; // Reduction dimension

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_jit(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i)
        A[i] = dist(rng);
    for (int i = 0; i < K * N; ++i)
        B[i] = dist(rng);

    reference_gemm(A.data(), B.data(), C_ref.data(), M, K, N);

    class GemmMicrokernelTest : public JitMicrokernelBase
    {
    public:
        GemmMicrokernelTest(int K, int lda, int ldb, int ldc)
            : K_(K), lda_(lda), ldb_(ldb), ldc_(ldc)
        {
            generate_code();
        }
        void call(const float *A, const float *B, float *C)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(A, B, C);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemm_microkernel_6x16(*this, rdi, rsi, rdx, K_, lda_, ldb_, ldc_);
            ret();
            ready();
        }
        int K_, lda_, ldb_, ldc_;
    };

    GemmMicrokernelTest kernel(K, K, N, N);
    kernel.call(A.data(), B.data(), C_jit.data());

    float max_err = compute_max_abs_error(C_jit.data(), C_ref.data(), M * N);
    float rms_err = compute_rms_error(C_jit.data(), C_ref.data(), M * N);

    EXPECT_LT(max_err, 1e-4f) << "Max error too large for 6x16 microkernel";
    EXPECT_LT(rms_err, 1e-5f) << "RMS error too large for 6x16 microkernel";
}

TEST_F(Test__JitWoProjectionOptimized, GEMM_Microkernel_6x16_LargeK)
{
    // Test with larger K (Qwen d_model size)
    constexpr int M = 6;
    constexpr int N = 16;
    constexpr int K = 896;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_jit(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < M * K; ++i)
        A[i] = dist(rng);
    for (int i = 0; i < K * N; ++i)
        B[i] = dist(rng);

    reference_gemm(A.data(), B.data(), C_ref.data(), M, K, N);

    class GemmMicrokernelTest : public JitMicrokernelBase
    {
    public:
        GemmMicrokernelTest(int K, int lda, int ldb, int ldc)
            : K_(K), lda_(lda), ldb_(ldb), ldc_(ldc)
        {
            generate_code();
        }
        void call(const float *A, const float *B, float *C)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(A, B, C);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemm_microkernel_6x16(*this, rdi, rsi, rdx, K_, lda_, ldb_, ldc_);
            ret();
            ready();
        }
        int K_, lda_, ldb_, ldc_;
    };

    GemmMicrokernelTest kernel(K, K, N, N);
    kernel.call(A.data(), B.data(), C_jit.data());

    float max_err = compute_max_abs_error(C_jit.data(), C_ref.data(), M * N);
    EXPECT_LT(max_err, 1e-3f) << "Max error too large for 6x16 with K=896";
}

// ============================================================================
// High-Level Dispatcher Tests
// ============================================================================

TEST_F(Test__JitWoProjectionOptimized, Dispatcher_DecodeUsesGEMV)
{
    // M=1 should dispatch to GEMV path
    constexpr int M = 1;
    constexpr int K = 128;
    constexpr int N = 128;

    std::vector<float> context(M * K);
    std::vector<float> Wo(K * N);
    std::vector<float> output_jit(M * N, 0.0f);
    std::vector<float> output_ref(M * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : context)
        v = dist(rng);
    for (auto &v : Wo)
        v = dist(rng);

    reference_gemm(context.data(), Wo.data(), output_ref.data(), M, K, N);

    class DispatchKernel : public JitMicrokernelBase
    {
    public:
        DispatchKernel(int M, int K, int N) : M_(M), K_(K), N_(N)
        {
            generate_code();
        }
        void call(const float *ctx, const float *Wo, float *out)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(ctx, Wo, out);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_wo_projection_optimized(*this, rdi, rsi, rdx, M_, K_, N_);
            ret();
            ready();
        }
        int M_, K_, N_;
    };

    DispatchKernel kernel(M, K, N);
    kernel.call(context.data(), Wo.data(), output_jit.data());

    float max_err = compute_max_abs_error(output_jit.data(), output_ref.data(), M * N);
    EXPECT_LT(max_err, 1e-4f) << "Dispatcher decode path error too large";
}

// ============================================================================
// Performance Sanity Check (not a benchmark, just ensures it's fast enough)
// ============================================================================

TEST_F(Test__JitWoProjectionOptimized, DISABLED_PerformanceSanity_GEMV_896)
{
    // This test is disabled by default but can be enabled for manual perf checks
    constexpr int K = 896;
    constexpr int N = 896;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    std::vector<float> context(K, 0.5f);
    std::vector<float> Wo(K * N, 0.1f);
    std::vector<float> output(N, 0.0f);

    class GemvKernel : public JitMicrokernelBase
    {
    public:
        GemvKernel(int K, int N) : K_(K), N_(N)
        {
            generate_code();
        }
        void call(const float *context, const float *Wo, float *output)
        {
            auto fn = getCode<void (*)(const float *, const float *, float *)>();
            fn(context, Wo, output);
        }

    private:
        void generate_code()
        {
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_4x64_fp32(*this, rdi, rsi, rdx, K_, N_);
            ret();
            ready();
        }
        int K_, N_;
    };

    GemvKernel kernel(K, N);

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
    {
        kernel.call(context.data(), Wo.data(), output.data());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i)
    {
        kernel.call(context.data(), Wo.data(), output.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ns = std::chrono::duration<double, std::nano>(end - start).count();
    double per_call_us = total_ns / ITERS / 1000.0;

    // 2 * K * N FLOPs per call
    double flops_per_call = 2.0 * K * N;
    double gflops = (flops_per_call * ITERS) / (total_ns / 1e9) / 1e9;

    std::cout << "\n=== GEMV 896x896 Performance ===" << std::endl;
    std::cout << "Per-call: " << std::fixed << std::setprecision(2) << per_call_us << " µs" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;

    // Sanity check: should be at least 10 GFLOP/s (target is ~50)
    EXPECT_GT(gflops, 10.0) << "Performance is unexpectedly low";
}
