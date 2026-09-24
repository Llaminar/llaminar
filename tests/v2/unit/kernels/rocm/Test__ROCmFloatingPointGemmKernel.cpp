/**
 * @file Test__ROCmFloatingPointGemmKernel.cpp
 * @brief Hardware integration proofs for floating GEMM and projection lifetime.
 *
 * Tests the ROCmFloatingPointGemmKernel which wraps hipBLAS for FP32/FP16/BF16
 * GEMM operations on AMD GPUs (MI50, MI100, MI250, etc.)
 * Retiring expert adapters must leave context-owned library resources alive
 * and must never drain unrelated inference streams.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

#ifdef HAVE_ROCM

#include "kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "kernels/rocm/gemm/HipBLASGemmKernel.h"
#include "backends/DeviceId.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <future>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>

using namespace llaminar2;
using namespace llaminar2::rocm;

namespace
{
    /** @brief Materialize the kernel's exact declared scratch before test execution. */
    std::unique_ptr<DeviceWorkspaceManager> bindDeclaredWorkspace(
        IWorkspaceConsumer &kernel, DeviceId device, int m = 1, int n = 1, int k = 1)
    {
        const auto requirements = kernel.getWorkspaceRequirements(m, n, k);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            device, requirements.total_bytes_with_alignment());
        if (!workspace->allocate(requirements))
            throw std::runtime_error("Could not materialize declared test hipBLAS workspace");
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

    /** @brief Test-only gate that keeps one HIP stream occupied. */
    class HostBlockedHipStream final
    {
    public:
        /** @brief Create a nonblocking stream and park a host function on it. */
        HostBlockedHipStream()
        {
            if (hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) !=
                hipSuccess)
            {
                throw std::runtime_error(
                    "Could not create adversarial HIP stream");
            }
            if (hipLaunchHostFunc(
                    stream_,
                    [](void *opaque)
                    {
                        auto *self = static_cast<HostBlockedHipStream *>(opaque);
                        self->entered_.store(true, std::memory_order_release);
                        while (!self->release_.load(std::memory_order_acquire))
                            std::this_thread::yield();
                    },
                    this) != hipSuccess)
            {
                (void)hipStreamDestroy(stream_);
                stream_ = nullptr;
                throw std::runtime_error(
                    "Could not enqueue adversarial HIP host gate");
            }

            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(5);
            while (!entered_.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::yield();
            }
            if (!entered_.load(std::memory_order_acquire))
            {
                release_.store(true, std::memory_order_release);
                (void)hipStreamSynchronize(stream_);
                (void)hipStreamDestroy(stream_);
                stream_ = nullptr;
                throw std::runtime_error(
                    "Adversarial HIP host gate did not start");
            }
        }

        /** @brief Release the gate before destroying its stream. */
        ~HostBlockedHipStream()
        {
            release();
            if (stream_)
            {
                (void)hipStreamDestroy(stream_);
            }
        }

        HostBlockedHipStream(const HostBlockedHipStream &) = delete;
        HostBlockedHipStream &operator=(const HostBlockedHipStream &) = delete;

        /** @return Exact non-default stream held behind the host gate. */
        [[nodiscard]] hipStream_t get() const noexcept { return stream_; }

        /** @brief Release the adversarial work before joining a retirement thread. */
        void release() noexcept
        {
            release_.store(true, std::memory_order_release);
            if (stream_)
                (void)hipStreamSynchronize(stream_);
        }

    private:
        hipStream_t stream_ = nullptr;
        std::atomic<bool> entered_{false};
        std::atomic<bool> release_{false};
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name);
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            setenv(name_.c_str(), value, 1);
        }

        ~ScopedEnv()
        {
            if (had_old_value_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        bool had_old_value_ = false;
        std::string old_value_;
    };
}

// ============================================================================
// Test Fixture
// ============================================================================

class Test__ROCmFloatingPointGemmKernel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if ROCm device is available
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No ROCm devices available";
        }

        rocm_device_id_ = 0;
        (void)hipSetDevice(rocm_device_id_);
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
            hipSuccess);

        // Get device properties
        hipDeviceProp_t props;
        (void)hipGetDeviceProperties(&props, rocm_device_id_);
        LOG_INFO("[Test] Using ROCm device: " << props.name << " (gfx" << props.gcnArchName << ")");
    }

    void TearDown() override
    {
        if (stream_)
        {
            (void)hipStreamSynchronize(stream_);
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Reference CPU GEMM for validation: C = A @ B^T (row-major)
    void reference_gemm(const float *A, const float *B, float *C,
                        int M, int N, int K, bool transpose_B = true)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    float a_val = A[m * K + k];
                    float b_val = transpose_B ? B[n * K + k] : B[k * N + n];
                    sum += a_val * b_val;
                }
                C[m * N + n] = sum;
            }
        }
    }

    // Allocate GPU memory and copy data
    float *allocate_and_copy_to_gpu(const std::vector<float> &host_data)
    {
        float *d_ptr = nullptr;
        (void)hipMalloc(&d_ptr, host_data.size() * sizeof(float));
        (void)hipMemcpy(d_ptr, host_data.data(), host_data.size() * sizeof(float), hipMemcpyHostToDevice);
        return d_ptr;
    }

    void copy_from_gpu(float *d_ptr, std::vector<float> &host_data)
    {
        // The producer stream is deliberately nonblocking.  Wait for that
        // exact stream before a synchronous host copy; legacy/default-stream
        // coupling must not provide hidden ordering in this harness.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        (void)hipMemcpy(host_data.data(), d_ptr, host_data.size() * sizeof(float), hipMemcpyDeviceToHost);
    }

    // Compute relative error
    float compute_relative_error(const std::vector<float> &ref, const std::vector<float> &actual)
    {
        float max_rel_err = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            float abs_err = std::abs(ref[i] - actual[i]);
            float denom = std::max(std::abs(ref[i]), 1e-6f);
            float rel_err = abs_err / denom;
            max_rel_err = std::max(max_rel_err, rel_err);
        }
        return max_rel_err;
    }

    // Compute cosine similarity: dot(a,b) / (||a|| * ||b||)
    float compute_cosine_similarity(const std::vector<float> &ref, const std::vector<float> &actual)
    {
        double dot = 0.0, norm_ref = 0.0, norm_actual = 0.0;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            dot += static_cast<double>(ref[i]) * static_cast<double>(actual[i]);
            norm_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            norm_actual += static_cast<double>(actual[i]) * static_cast<double>(actual[i]);
        }
        double denom = std::sqrt(norm_ref) * std::sqrt(norm_actual);
        return denom > 1e-12 ? static_cast<float>(dot / denom) : 0.0f;
    }

    int rocm_device_id_ = 0;
    hipStream_t stream_ = nullptr;
};

// ============================================================================
// HipBLASGemmKernel Tests (Low-level)
// ============================================================================

/**
 * @brief Floating expert retirement must not wait for unrelated HIP execution.
 *
 * Retain the allocation independently, as an overlay slab does, and destroy
 * only FP16/BF16/FP32 projection views while an unrelated stream is parked.
 * Always release that stream before joining the destructor thread, so a
 * regression reports a bounded failure rather than hanging the integration gate.
 */
TEST_F(Test__ROCmFloatingPointGemmKernel,
       FloatingExpertRetirementDoesNotWaitForUnrelatedStream)
{
    using Adapter = ROCmFloatingPointGemmKernel;
    void *weights = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&weights, 16 * 16 * sizeof(float)));
    std::vector<std::unique_ptr<Adapter>> projections;
    for (const auto precision : {Adapter::Precision::FP16,
                                 Adapter::Precision::BF16,
                                 Adapter::Precision::FP32})
        projections.push_back(std::make_unique<Adapter>(
            weights, 16, 16, 0, precision, std::shared_ptr<void>{}));

    HostBlockedHipStream blocked;
    std::promise<void> retired;
    auto completion = retired.get_future();
    std::thread retirement([&]
    {
        (void)hipSetDevice(0);
        projections.clear();
        retired.set_value();
    });
    const auto status = completion.wait_for(std::chrono::seconds(2));
    blocked.release();
    retirement.join();
    EXPECT_EQ(status, std::future_status::ready)
        << "Expert retirement waited for unrelated device work (private BLAS teardown)";
    ASSERT_EQ(hipSuccess, hipFree(weights));
}

/** @brief A shared context handle captures independent stream/workspace pairs exactly. */
TEST_F(Test__ROCmFloatingPointGemmKernel, ContextHandleCapturedConcurrentStreamsReplayExactly)
{
    constexpr int M = 8, N = 64, K = 128;
    HipBLASGemmKernel primary(DeviceId::rocm(rocm_device_id_));
    auto primary_workspace = bindDeclaredWorkspace(primary, DeviceId::rocm(rocm_device_id_));
    primary.bindStream(ExplicitGPUStream{stream_});
    std::mt19937 rng(0x5345);
    std::uniform_real_distribution<float> distribution(-0.2f, 0.2f);
    std::vector<float> a(M * K), b(N * K);
    for (auto &value : a) value = distribution(rng);
    for (auto &value : b) value = distribution(rng);
    float *d_a = nullptr, *d_b = nullptr, *d_c0 = nullptr, *d_c1 = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&d_a, a.size() * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_b, b.size() * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_c0, M * N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_c1, M * N * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(d_a, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice, stream_));
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(d_b, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice, stream_));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
    hipStream_t side = nullptr;
    hipEvent_t fork = nullptr, join = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags(&fork, hipEventDisableTiming));
    ASSERT_EQ(hipSuccess, hipEventCreateWithFlags(&join, hipEventDisableTiming));
    HipBLASGemmKernel secondary(DeviceId::rocm(rocm_device_id_));
    const auto requirements = secondary.getWorkspaceRequirements(M, N, K);
    DeviceWorkspaceManager side_workspace(DeviceId::rocm(0), requirements.total_bytes_with_alignment());
    ASSERT_TRUE(side_workspace.allocate(requirements));
    secondary.bindWorkspace(&side_workspace);
    secondary.bindStream(ExplicitGPUStream{side});

    // Warm the identical library path once before recording. Save its exact
    // result, then prove replay has not mixed either stream's mutable bindings.
    ASSERT_TRUE(primary.execute(d_a, d_b, d_c0, M, N, K, false, false));
    ASSERT_TRUE(secondary.execute(d_a, d_b, d_c1, M, N, K, false, false));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(side));
    std::vector<float> expected(M * N), actual(M * N);
    ASSERT_EQ(hipSuccess, hipMemcpy(expected.data(), d_c0, expected.size() * sizeof(float), hipMemcpyDeviceToHost));

    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
    ASSERT_EQ(hipSuccess, hipEventRecord(fork, stream_));
    ASSERT_EQ(hipSuccess, hipStreamWaitEvent(side, fork, 0));
    ASSERT_TRUE(primary.execute(d_a, d_b, d_c0, M, N, K, false, false));
    ASSERT_TRUE(secondary.execute(d_a, d_b, d_c1, M, N, K, false, false));
    ASSERT_EQ(hipSuccess, hipEventRecord(join, side));
    ASSERT_EQ(hipSuccess, hipStreamWaitEvent(stream_, join, 0));
    ASSERT_EQ(hipSuccess, hipStreamEndCapture(stream_, &graph));
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 20; ++replay)
    {
        ASSERT_EQ(hipSuccess, hipMemsetAsync(d_c0, 0x7f, expected.size() * sizeof(float), stream_));
        ASSERT_EQ(hipSuccess, hipMemsetAsync(d_c1, 0x7f, expected.size() * sizeof(float), stream_));
        ASSERT_EQ(hipSuccess, hipGraphLaunch(executable, stream_));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream_));
        for (auto *output : {d_c0, d_c1})
        {
            ASSERT_EQ(hipSuccess, hipMemcpy(actual.data(), output, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
            EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)))
                << "replay=" << replay;
        }
    }
    ASSERT_EQ(hipSuccess, hipGraphExecDestroy(executable));
    ASSERT_EQ(hipSuccess, hipGraphDestroy(graph));
    ASSERT_EQ(hipSuccess, hipEventDestroy(join));
    ASSERT_EQ(hipSuccess, hipEventDestroy(fork));
    ASSERT_EQ(hipSuccess, hipStreamDestroy(side));
    ASSERT_EQ(hipSuccess, hipFree(d_c1));
    ASSERT_EQ(hipSuccess, hipFree(d_c0));
    ASSERT_EQ(hipSuccess, hipFree(d_b));
    ASSERT_EQ(hipSuccess, hipFree(d_a));
}


TEST_F(Test__ROCmFloatingPointGemmKernel, HipBLASGemmKernel_SmallMatrix)
{
    // Small 4x4 matrix test
    const int M = 4, N = 4, K = 4;

    // Initialize test data
    std::vector<float> h_A(M * K), h_B(N * K), h_C(M * N, 0.0f), h_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : h_A)
        v = dist(rng);
    for (auto &v : h_B)
        v = dist(rng);

    // Compute reference (C = A @ B^T)
    reference_gemm(h_A.data(), h_B.data(), h_ref.data(), M, N, K, true);

    // GPU computation
    float *d_A = allocate_and_copy_to_gpu(h_A);
    float *d_B = allocate_and_copy_to_gpu(h_B);
    float *d_C = allocate_and_copy_to_gpu(h_C);

    HipBLASGemmKernel kernel(DeviceId::rocm(rocm_device_id_));
    auto workspace = bindDeclaredWorkspace(kernel, DeviceId::rocm(rocm_device_id_));
    kernel.bindStream(ExplicitGPUStream{stream_});
    ASSERT_TRUE(kernel.execute(d_A, d_B, d_C, M, N, K, false, true));

    copy_from_gpu(d_C, h_C);

    // Validate
    float max_rel_err = compute_relative_error(h_ref, h_C);
    float cosine_sim = compute_cosine_similarity(h_ref, h_C);
    LOG_INFO("[Test] Small matrix - max relative error: " << max_rel_err << ", cosine similarity: " << cosine_sim);
    EXPECT_LT(max_rel_err, 1e-5f);
    EXPECT_GT(cosine_sim, 0.9999f); // Expect near-perfect alignment

    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, HipBLASGemmKernel_Qwen05B_Sizes)
{
    // Test with Qwen 0.5B typical sizes
    // FFN: [seq_len, hidden] @ [intermediate, hidden]^T = [seq_len, intermediate]
    const int M = 16;   // Batch/seq_len
    const int N = 4864; // Intermediate dim (Qwen 0.5B)
    const int K = 896;  // Hidden dim (Qwen 0.5B)

    std::vector<float> h_A(M * K), h_B(N * K), h_C(M * N, 0.0f), h_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &v : h_A)
        v = dist(rng);
    for (auto &v : h_B)
        v = dist(rng);

    reference_gemm(h_A.data(), h_B.data(), h_ref.data(), M, N, K, true);

    float *d_A = allocate_and_copy_to_gpu(h_A);
    float *d_B = allocate_and_copy_to_gpu(h_B);
    float *d_C = allocate_and_copy_to_gpu(h_C);

    HipBLASGemmKernel kernel(DeviceId::rocm(rocm_device_id_));
    auto workspace = bindDeclaredWorkspace(kernel, DeviceId::rocm(rocm_device_id_));
    kernel.bindStream(ExplicitGPUStream{stream_});
    ASSERT_TRUE(kernel.execute(d_A, d_B, d_C, M, N, K, false, true));

    copy_from_gpu(d_C, h_C);

    float max_rel_err = compute_relative_error(h_ref, h_C);
    float cosine_sim = compute_cosine_similarity(h_ref, h_C);
    LOG_INFO("[Test] Qwen 0.5B sizes - max relative error: " << max_rel_err << ", cosine similarity: " << cosine_sim);
    // Large matrix GEMM accumulates rounding errors - 10% tolerance is reasonable
    EXPECT_LT(max_rel_err, 0.1f);
    EXPECT_GT(cosine_sim, 0.999f); // Expect high alignment for GEMM

    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, HipBLASGemmKernel_Qwen14B_Sizes)
{
    // Test with Qwen 14B typical sizes (stress test)
    // Attention projection: [seq_len, hidden] @ [hidden, hidden]^T
    const int M = 32;   // Batch/seq_len
    const int N = 5120; // Hidden dim (Qwen 14B)
    const int K = 5120; // Hidden dim (Qwen 14B)

    std::vector<float> h_A(M * K), h_B(N * K), h_C(M * N, 0.0f), h_ref(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &v : h_A)
        v = dist(rng);
    for (auto &v : h_B)
        v = dist(rng);

    reference_gemm(h_A.data(), h_B.data(), h_ref.data(), M, N, K, true);

    float *d_A = allocate_and_copy_to_gpu(h_A);
    float *d_B = allocate_and_copy_to_gpu(h_B);
    float *d_C = allocate_and_copy_to_gpu(h_C);

    HipBLASGemmKernel kernel(DeviceId::rocm(rocm_device_id_));
    auto workspace = bindDeclaredWorkspace(kernel, DeviceId::rocm(rocm_device_id_));
    kernel.bindStream(ExplicitGPUStream{stream_});
    ASSERT_TRUE(kernel.execute(d_A, d_B, d_C, M, N, K, false, true));

    copy_from_gpu(d_C, h_C);

    float max_rel_err = compute_relative_error(h_ref, h_C);
    float cosine_sim = compute_cosine_similarity(h_ref, h_C);
    LOG_INFO("[Test] Qwen 14B sizes - max relative error: " << max_rel_err << ", cosine similarity: " << cosine_sim);
    // Large matrix GEMM accumulates rounding errors - 10% tolerance is reasonable
    EXPECT_LT(max_rel_err, 0.1f);
    EXPECT_GT(cosine_sim, 0.999f); // Expect high alignment for GEMM

    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, HipBLASGemmKernel_Performance)
{
    // Performance benchmark for Qwen 14B sizes
    const int M = 128;  // Larger batch for better GPU utilization
    const int N = 5120; // Qwen 14B hidden
    const int K = 5120;

    std::vector<float> h_A(M * K), h_B(N * K), h_C(M * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &v : h_A)
        v = dist(rng);
    for (auto &v : h_B)
        v = dist(rng);

    float *d_A = allocate_and_copy_to_gpu(h_A);
    float *d_B = allocate_and_copy_to_gpu(h_B);
    float *d_C = allocate_and_copy_to_gpu(h_C);

    HipBLASGemmKernel kernel(DeviceId::rocm(rocm_device_id_));
    auto workspace = bindDeclaredWorkspace(kernel, DeviceId::rocm(rocm_device_id_));
    kernel.bindStream(ExplicitGPUStream{stream_});

    // Warmup
    kernel.execute(d_A, d_B, d_C, M, N, K, false, true);
    (void)hipDeviceSynchronize();

    // Benchmark
    const int num_iters = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; ++i)
    {
        kernel.execute(d_A, d_B, d_C, M, N, K, false, true);
    }
    (void)hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Calculate GFLOPS
    double flops_per_iter = 2.0 * M * N * K; // 2 * M * N * K for GEMM
    double total_flops = flops_per_iter * num_iters;
    double gflops = total_flops / (elapsed_ms * 1e6); // GFLOPS

    LOG_INFO("[Test] hipBLAS GEMM Performance:");
    LOG_INFO("  Matrix sizes: M=" << M << " N=" << N << " K=" << K);
    LOG_INFO("  Iterations: " << num_iters);
    LOG_INFO("  Time: " << elapsed_ms << " ms total, " << (elapsed_ms / num_iters) << " ms/iter");
    LOG_INFO("  Performance: " << gflops << " GFLOPS");

    // Note: No performance assertion - GFLOPS varies significantly when
    // running in parallel with other GPU tests due to resource contention

    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);
}

// ============================================================================
// ROCmFloatingPointGemmKernel Tests (ITensorGemm interface)
// ============================================================================

TEST_F(Test__ROCmFloatingPointGemmKernel, TensorInterface_Basic)
{
    const size_t M = 16, N = 256, K = 128;

    // Create weight tensor
    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K}); // [N, K] for transpose
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    float *w_data = weights->mutable_data();
    for (size_t i = 0; i < N * K; ++i)
        w_data[i] = dist(rng);

    // Upload weights to GPU
    ASSERT_TRUE(weights->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    // Create kernel
    ROCmFloatingPointGemmKernel kernel(weights.get(), rocm_device_id_);
    auto workspace = bindDeclaredWorkspace(kernel, DeviceId::rocm(rocm_device_id_), M, N, K);
    kernel.setGPUStream(stream_);

    // Create input/output tensors
    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});

    float *in_data = input->mutable_data();
    for (size_t i = 0; i < M * K; ++i)
        in_data[i] = dist(rng);

    // Upload to GPU
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    // Execute GEMM
    ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get()));

    // Verify output is not all zeros (sanity check)
    TransferEngine::publishCurrentDeviceWrite(output, stream_);
    const float *out_data = output->data();

    float sum = 0.0f;
    for (size_t i = 0; i < M * N; ++i)
        sum += std::abs(out_data[i]);

    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";
    LOG_INFO("[Test] TensorInterface basic test passed, output sum=" << sum);
}

/**
 * @brief Prove one cached hipBLAS handle cannot leak another kernel's stream.
 *
 * Both wrappers intentionally borrow the same context-owned hipBLAS handle. The
 * second wrapper binds a stream parked behind a test-only host gate before the
 * first wrapper submits. Correct code carries the first wrapper's exact stream
 * into the locked bind-and-submit transaction, so its result is observable
 * without releasing the unrelated stream.
 */
TEST_F(
    Test__ROCmFloatingPointGemmKernel,
    SharedHipBLASHandleKeepsEachProjectionOnItsExactStream)
{
    constexpr std::size_t M = 8;
    constexpr std::size_t N = 128;
    constexpr std::size_t K = 128;
    const DeviceId device = DeviceId::rocm(rocm_device_id_);

    auto input = std::make_unique<FP32Tensor>(
        std::vector<std::size_t>{M, K});
    auto weights_a = std::make_unique<FP32Tensor>(
        std::vector<std::size_t>{N, K});
    auto weights_b = std::make_unique<FP32Tensor>(
        std::vector<std::size_t>{N, K});
    auto output = std::make_unique<FP32Tensor>(
        std::vector<std::size_t>{M, N});

    std::mt19937 rng(0x51deu);
    std::uniform_real_distribution<float> distribution(-0.2f, 0.2f);
    for (std::size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = distribution(rng);
    for (std::size_t i = 0; i < N * K; ++i)
    {
        weights_a->mutable_data()[i] = distribution(rng);
        weights_b->mutable_data()[i] = distribution(rng);
    }
    std::fill(
        output->mutable_data(),
        output->mutable_data() + M * N,
        0.0f);

    std::vector<float> reference(M * N);
    reference_gemm(
        input->data(),
        weights_a->data(),
        reference.data(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        true);

    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(weights_a->ensureOnDevice(device));
    ASSERT_TRUE(weights_b->ensureOnDevice(device));
    ASSERT_TRUE(output->ensureOnDevice(device));

    ROCmFloatingPointGemmKernel kernel_a(
        weights_a.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel kernel_b(
        weights_b.get(), rocm_device_id_);
    EXPECT_FALSE(kernel_a.multiply_tensor(input.get(), output.get()))
        << "An unbound floating GEMM must not inherit a cached handle stream";

    auto workspace_a = bindDeclaredWorkspace(kernel_a, device, M, N, K);
    auto workspace_b = bindDeclaredWorkspace(kernel_b, device, M, N, K);

    HostBlockedHipStream blocked;
    kernel_a.setGPUStream(stream_);
    kernel_b.setGPUStream(blocked.get());
    ASSERT_TRUE(kernel_a.multiply_tensor(input.get(), output.get()));

    std::vector<float> actual(M * N);
    ASSERT_EQ(
        hipMemcpyAsync(
            actual.data(),
            output->gpu_data_ptr(),
            actual.size() * sizeof(float),
            hipMemcpyDeviceToHost,
            stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

    EXPECT_GT(compute_cosine_similarity(reference, actual), 0.99999f);
    EXPECT_LT(compute_relative_error(reference, actual), 1.0e-3f);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, BatchedFusedProjectionWorkspaceNamesMergeCanonical)
{
    const size_t M = 2, N = 8, K = 16;

    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});

    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);

    WorkspaceRequirements reqs;
    reqs.merge(alpha_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    reqs.merge(beta_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));

    int a_ptr_buffers = 0;
    int b_ptr_buffers = 0;
    int c_ptr_buffers = 0;
    int old_slice_named_buffers = 0;
    const std::vector<std::string> old_prefixes = {
        std::string(GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS) + "_",
        std::string(GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS) + "_",
        std::string(GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS) + "_",
    };

    for (const auto &buf : reqs.buffers)
    {
        if (buf.name == GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS)
            ++a_ptr_buffers;
        if (buf.name == GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS)
            ++b_ptr_buffers;
        if (buf.name == GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS)
            ++c_ptr_buffers;
        for (const auto &prefix : old_prefixes)
        {
            if (buf.name.rfind(prefix, 0) == 0)
                ++old_slice_named_buffers;
        }
    }

    EXPECT_EQ(a_ptr_buffers, 1);
    EXPECT_EQ(b_ptr_buffers, 1);
    EXPECT_EQ(c_ptr_buffers, 1);
    EXPECT_EQ(old_slice_named_buffers, 0)
        << "FP32 batched pointer arrays must not use per-kernel names that churn graph workspace";
}

TEST_F(Test__ROCmFloatingPointGemmKernel, GraphCapturedBatchedFusedProjectionAlphaBetaM2MatchesReference)
{
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const size_t M = 2, N = 32, K = 256;

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto output_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
    auto output_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    for (size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist(rng);
    for (size_t i = 0; i < N * K; ++i)
    {
        weights_alpha->mutable_data()[i] = dist(rng);
        weights_beta->mutable_data()[i] = dist(rng);
    }

    std::vector<float> ref_alpha(M * N);
    std::vector<float> ref_beta(M * N);
    reference_gemm(input->data(), weights_alpha->data(), ref_alpha.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);
    reference_gemm(input->data(), weights_beta->data(), ref_beta.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);
    ASSERT_TRUE(alpha_kernel.supports_fused_projection());
    ASSERT_TRUE(beta_kernel.supports_fused_projection());

    WorkspaceRequirements reqs;
    reqs.merge(alpha_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    reqs.merge(beta_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(reqs));
    alpha_kernel.bindWorkspace(&workspace);
    beta_kernel.bindWorkspace(&workspace);
    ASSERT_TRUE(alpha_kernel.hasWorkspace());
    ASSERT_TRUE(beta_kernel.hasWorkspace());

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    alpha_kernel.setGPUStream(stream);
    beta_kernel.setGPUStream(stream);

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha"},
        {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta"}};

    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemsetAsync(output_alpha->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(output_beta->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
    TransferEngine::publishCurrentDeviceWrite(output_beta, stream);
    const float *actual_alpha = output_alpha->data();
    const float *actual_beta = output_beta->data();
    std::vector<float> got_alpha(actual_alpha, actual_alpha + M * N);
    std::vector<float> got_beta(actual_beta, actual_beta + M * N);

    const float alpha_cosine = compute_cosine_similarity(ref_alpha, got_alpha);
    const float beta_cosine = compute_cosine_similarity(ref_beta, got_beta);
    EXPECT_GT(alpha_cosine, 0.9999f);
    EXPECT_GT(beta_cosine, 0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    const auto small_n_route = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "kernel" &&
                   record.name == "rocm_fp32_small_n_batched_projection_calls" &&
                   record.device == "rocm:0" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "2" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "32" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "256" &&
                   record.tags.count("batch") != 0 &&
                   record.tags.at("batch") == "2";
        });
    ASSERT_NE(small_n_route, records.end())
        << "Graph-captured M=2 alpha/beta projections should use the ROCm small-N FP32 batched route";
    EXPECT_GE(small_n_route->value, 1.0);

    const auto hipblas_route = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "kernel" &&
                   record.name == "rocm_fp32_batched_projection_calls" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "2" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "32" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "256" &&
                   record.tags.count("batch") != 0 &&
                   record.tags.at("batch") == "2";
        });
    EXPECT_EQ(hipblas_route, records.end())
        << "Small-N graph-captured FP32 projection shapes should not pay hipBLAS batched launch overhead";

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, BatchedFusedProjectionVerifierRowsM234MatchSerialDecodeRows)
{
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const size_t N = 48, K = 256;
    const std::array<int, 3> verifier_rows = {2, 3, 4};

    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});

    std::mt19937 rng(177);
    std::uniform_real_distribution<float> input_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);
    for (size_t i = 0; i < N * K; ++i)
    {
        weights_alpha->mutable_data()[i] = weight_dist(rng);
        weights_beta->mutable_data()[i] = weight_dist(rng);
    }

    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);
    ASSERT_TRUE(alpha_kernel.supports_fused_projection());
    ASSERT_TRUE(beta_kernel.supports_fused_projection());

    WorkspaceRequirements reqs;
    reqs.merge(alpha_kernel.getWorkspaceRequirements(4, static_cast<int>(N), static_cast<int>(K)));
    reqs.merge(beta_kernel.getWorkspaceRequirements(4, static_cast<int>(N), static_cast<int>(K)));
    DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(reqs));
    alpha_kernel.bindWorkspace(&workspace);
    beta_kernel.bindWorkspace(&workspace);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    alpha_kernel.setGPUStream(stream);
    beta_kernel.setGPUStream(stream);

    for (int M : verifier_rows)
    {
        SCOPED_TRACE(std::string("ROCm FP32 M=") + std::to_string(M));
        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), K});
        for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
            input->mutable_data()[i] = input_dist(rng);

        auto output_alpha = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), N});
        auto output_beta = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), N});
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
        ASSERT_TRUE(output_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
        ASSERT_TRUE(output_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

        std::vector<ITensorGemm::TensorProjectionDesc> grouped_projections = {
            {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha_grouped"},
            {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta_grouped"}};

        ASSERT_TRUE(alpha_kernel.multiply_fused_verifier_rows_decode_equivalent(
            input.get(), grouped_projections, M, static_cast<int>(K), nullptr, &workspace))
            << "ROCm FP32 grouped verifier projection failed";
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
        TransferEngine::publishCurrentDeviceWrite(output_beta, stream);

        for (int row = 0; row < M; ++row)
        {
            auto row_input = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, K});
            std::memcpy(
                row_input->mutable_data(),
                input->data() + static_cast<size_t>(row) * K,
                K * sizeof(float));
            auto alpha_serial = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, N});
            auto beta_serial = std::make_unique<FP32Tensor>(
                std::vector<size_t>{size_t{1}, N});
            ASSERT_TRUE(row_input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
            ASSERT_TRUE(alpha_serial->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
            ASSERT_TRUE(beta_serial->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

            std::vector<ITensorGemm::TensorProjectionDesc> serial_projections = {
                {&alpha_kernel, alpha_serial.get(), static_cast<int>(N), nullptr, "alpha_serial"},
                {&beta_kernel, beta_serial.get(), static_cast<int>(N), nullptr, "beta_serial"}};
            ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
                row_input.get(), serial_projections, 1, static_cast<int>(K), nullptr, &workspace))
                << "ROCm FP32 serial decode projection failed for row " << row;
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            TransferEngine::publishCurrentDeviceWrite(alpha_serial, stream);
            TransferEngine::publishCurrentDeviceWrite(beta_serial, stream);

            EXPECT_EQ(
                std::memcmp(
                    output_alpha->data() + static_cast<size_t>(row) * N,
                    alpha_serial->data(),
                    N * sizeof(float)),
                0)
                << "ROCm FP32 alpha grouped verifier row must be bitwise serial-decode equivalent";
            EXPECT_EQ(
                std::memcmp(
                    output_beta->data() + static_cast<size_t>(row) * N,
                    beta_serial->data(),
                    N * sizeof(float)),
                0)
                << "ROCm FP32 beta grouped verifier row must be bitwise serial-decode equivalent";
        }
    }

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_fp32_small_n_batched_projection_calls"});
    uint64_t grouped_small_n_calls = 0;
    for (const auto &record : records)
    {
        if (record.kind == PerfStatRecord::Kind::Counter &&
            record.domain == "kernel" &&
            record.name == "rocm_fp32_small_n_batched_projection_calls" &&
            record.tags.at("n") == std::to_string(N) &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.at("batch") == "2")
        {
            const std::string &m_tag = record.tags.at("m");
            if (m_tag == "2" || m_tag == "3" || m_tag == "4")
                grouped_small_n_calls += record.count;
        }
    }
    EXPECT_EQ(grouped_small_n_calls, verifier_rows.size())
        << "ROCm FP32 verifier rows must use the small-N grouped batched projection route for M=2/3/4";

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    PerfStatsCollector::reset();
}

TEST_F(Test__ROCmFloatingPointGemmKernel, FP16BF16VerifierRowsM234MatchSerialDecodeRows)
{
    /**
     * ROCm FP16/BF16 floating weights participate in verifier publication with
     * FP32 hidden rows and FP32 outputs.  The grouped M=2..4 hook and the normal
     * M=1 multiply_tensor decode entry point must therefore share the same
     * fixed-order fp32x16 device kernel.
     */
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const size_t N = 80, K = 192;
    const std::array<int, 3> verifier_rows = {2, 3, 4};

    auto run_case = [&](bool bf16)
    {
        const char *dtype_tag = bf16 ? "bf16" : "fp16";
        std::mt19937 rng(bf16 ? 533 : 431);
        std::uniform_real_distribution<float> input_dist(-0.75f, 0.75f);
        std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);

        std::vector<float> weight_alpha_fp32(N * K);
        std::vector<float> weight_beta_fp32(N * K);
        for (float &value : weight_alpha_fp32)
            value = weight_dist(rng);
        for (float &value : weight_beta_fp32)
            value = weight_dist(rng);

        std::unique_ptr<TensorBase> weights_alpha;
        std::unique_ptr<TensorBase> weights_beta;
        ROCmFloatingPointGemmKernel::Precision precision;
        if (bf16)
        {
            auto alpha = std::make_unique<BF16Tensor>(std::vector<size_t>{N, K});
            auto beta = std::make_unique<BF16Tensor>(std::vector<size_t>{N, K});
            alpha->from_fp32(weight_alpha_fp32.data(), weight_alpha_fp32.size());
            beta->from_fp32(weight_beta_fp32.data(), weight_beta_fp32.size());
            weights_alpha = std::move(alpha);
            weights_beta = std::move(beta);
            precision = ROCmFloatingPointGemmKernel::Precision::BF16;
        }
        else
        {
            auto alpha = std::make_unique<FP16Tensor>(std::vector<size_t>{N, K});
            auto beta = std::make_unique<FP16Tensor>(std::vector<size_t>{N, K});
            alpha->from_fp32(weight_alpha_fp32.data(), weight_alpha_fp32.size());
            beta->from_fp32(weight_beta_fp32.data(), weight_beta_fp32.size());
            weights_alpha = std::move(alpha);
            weights_beta = std::move(beta);
            precision = ROCmFloatingPointGemmKernel::Precision::FP16;
        }

        ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
        ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

        ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_, precision);
        ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_, precision);
        ASSERT_TRUE(alpha_kernel.supports_fused_projection())
            << "FP16/BF16 weights must advertise the installed fixed-order grouped projection path";

        WorkspaceRequirements reqs;
        reqs.merge(alpha_kernel.getWorkspaceRequirements(4, static_cast<int>(N), static_cast<int>(K)));
        reqs.merge(beta_kernel.getWorkspaceRequirements(4, static_cast<int>(N), static_cast<int>(K)));
        DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(reqs));
        alpha_kernel.bindWorkspace(&workspace);
        beta_kernel.bindWorkspace(&workspace);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        alpha_kernel.setGPUStream(stream);
        beta_kernel.setGPUStream(stream);
        // Weight upload has a published producer event; join it to the exact
        // consumer stream instead of relying on library setup to drain the GPU.
        TransferEngine::requireDeviceInput(weights_alpha.get(), DeviceId::rocm(rocm_device_id_), stream);
        TransferEngine::requireDeviceInput(weights_beta.get(), DeviceId::rocm(rocm_device_id_), stream);

        for (int M : verifier_rows)
        {
            SCOPED_TRACE(std::string("ROCm ") + dtype_tag + " M=" + std::to_string(M));
            auto input = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), K});
            for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
                input->mutable_data()[i] = input_dist(rng);

            auto output_alpha = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), N});
            auto output_beta = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), N});
            ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    input.get(), DeviceId::rocm(rocm_device_id_), stream));
            ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    output_alpha.get(), DeviceId::rocm(rocm_device_id_), stream));
            ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    output_beta.get(), DeviceId::rocm(rocm_device_id_), stream));

            std::vector<ITensorGemm::TensorProjectionDesc> grouped_projections = {
                {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha_grouped"},
                {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta_grouped"}};

            ASSERT_TRUE(alpha_kernel.multiply_fused_verifier_rows_decode_equivalent(
                input.get(), grouped_projections, M, static_cast<int>(K), nullptr, &workspace))
                << "ROCm " << dtype_tag << " grouped verifier projection failed";
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
            TransferEngine::publishCurrentDeviceWrite(output_beta, stream);

            for (int row = 0; row < M; ++row)
            {
                auto row_input = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, K});
                std::memcpy(
                    row_input->mutable_data(),
                    input->data() + static_cast<size_t>(row) * K,
                    K * sizeof(float));
                auto alpha_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, N});
                auto beta_serial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, N});
                ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    row_input.get(), DeviceId::rocm(rocm_device_id_), stream));
                ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    alpha_serial.get(), DeviceId::rocm(rocm_device_id_), stream));
                ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    beta_serial.get(), DeviceId::rocm(rocm_device_id_), stream));

                ASSERT_TRUE(alpha_kernel.multiply_tensor(
                    row_input.get(), alpha_serial.get(), 1, static_cast<int>(N), static_cast<int>(K),
                    true, 1.0f, 0.0f, nullptr, nullptr, -1, &workspace));
                ASSERT_TRUE(beta_kernel.multiply_tensor(
                    row_input.get(), beta_serial.get(), 1, static_cast<int>(N), static_cast<int>(K),
                    true, 1.0f, 0.0f, nullptr, nullptr, -1, &workspace));
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(alpha_serial, stream);
                TransferEngine::publishCurrentDeviceWrite(beta_serial, stream);

                EXPECT_EQ(
                    std::memcmp(
                        output_alpha->data() + static_cast<size_t>(row) * N,
                        alpha_serial->data(),
                        N * sizeof(float)),
                    0)
                    << "ROCm " << dtype_tag << " alpha grouped verifier row must be bitwise serial-decode equivalent";
                EXPECT_EQ(
                    std::memcmp(
                        output_beta->data() + static_cast<size_t>(row) * N,
                        beta_serial->data(),
                        N * sizeof(float)),
                    0)
                    << "ROCm " << dtype_tag << " beta grouped verifier row must be bitwise serial-decode equivalent";
            }
        }

        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    };

    run_case(false);
    run_case(true);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    uint64_t fp16_calls = 0;
    uint64_t bf16_calls = 0;
    for (const auto &record : records)
    {
        if (record.kind != PerfStatRecord::Kind::Counter ||
            record.domain != "kernel" ||
            record.name != "rocm_fp32x16_grouped_verifier_projection_calls" ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.at("k") != std::to_string(K) ||
            record.tags.at("projections") != "2")
        {
            continue;
        }
        const std::string &m_tag = record.tags.at("m");
        if (!(m_tag == "2" || m_tag == "3" || m_tag == "4"))
            continue;
        if (record.tags.at("dtype") == "fp16")
            fp16_calls += record.count;
        else if (record.tags.at("dtype") == "bf16")
            bf16_calls += record.count;
    }
    EXPECT_EQ(fp16_calls, verifier_rows.size())
        << PerfStatsCollector::summaryString({"kernel"});
    EXPECT_EQ(bf16_calls, verifier_rows.size())
        << PerfStatsCollector::summaryString({"kernel"});

    PerfStatsCollector::reset();
}

TEST_F(Test__ROCmFloatingPointGemmKernel, FloatingSwiGLUDownVerifierRowsM234MatchSerialDecodeRows)
{
    /**
     * The floating shared-expert verifier path needs a real grouped
     * SwiGLU/down implementation for every floating down-weight format.  This
     * test runs FP32, FP16, and BF16 down weights through grouped M=2..4 and
     * compares each row with the same decode-sized fused entry point at M=1.
     */
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const size_t N = 80, K = 192;
    const std::array<int, 3> verifier_rows = {2, 3, 4};

    auto run_case = [&](const char *dtype_tag, ROCmFloatingPointGemmKernel::Precision precision)
    {
        std::mt19937 rng(
            precision == ROCmFloatingPointGemmKernel::Precision::FP32 ? 631 :
            precision == ROCmFloatingPointGemmKernel::Precision::FP16 ? 733 : 839);
        std::uniform_real_distribution<float> activation_dist(-0.65f, 0.65f);
        std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);

        std::vector<float> weight_fp32(N * K);
        for (float &value : weight_fp32)
            value = weight_dist(rng);

        std::unique_ptr<TensorBase> weights_down;
        if (precision == ROCmFloatingPointGemmKernel::Precision::FP32)
        {
            auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
            std::memcpy(tensor->mutable_data(), weight_fp32.data(), weight_fp32.size() * sizeof(float));
            weights_down = std::move(tensor);
        }
        else if (precision == ROCmFloatingPointGemmKernel::Precision::FP16)
        {
            auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{N, K});
            tensor->from_fp32(weight_fp32.data(), weight_fp32.size());
            weights_down = std::move(tensor);
        }
        else
        {
            auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{N, K});
            tensor->from_fp32(weight_fp32.data(), weight_fp32.size());
            weights_down = std::move(tensor);
        }

        ASSERT_TRUE(weights_down->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
        ROCmFloatingPointGemmKernel down_kernel(weights_down.get(), rocm_device_id_, precision);

        WorkspaceRequirements reqs;
        reqs.merge(down_kernel.getWorkspaceRequirements(4, static_cast<int>(N), static_cast<int>(K)));
        DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(reqs));
        down_kernel.bindWorkspace(&workspace);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        down_kernel.setGPUStream(stream);
        // Weight upload has a published producer event; join it to the exact
        // consumer stream instead of relying on library setup to drain the GPU.
        TransferEngine::requireDeviceInput(weights_down.get(), DeviceId::rocm(rocm_device_id_), stream);

        for (int M : verifier_rows)
        {
            SCOPED_TRACE(std::string("ROCm ") + dtype_tag + " SwiGLU/down M=" + std::to_string(M));
            auto gate = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), K});
            auto up = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), K});
            for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
            {
                gate->mutable_data()[i] = activation_dist(rng);
                up->mutable_data()[i] = activation_dist(rng);
            }

            auto grouped_down = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), N});
            ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    gate.get(), DeviceId::rocm(rocm_device_id_), stream));
            ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    up.get(), DeviceId::rocm(rocm_device_id_), stream));
            ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    grouped_down.get(), DeviceId::rocm(rocm_device_id_), stream));

            ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                gate.get(), up.get(), grouped_down.get(),
                M, static_cast<int>(N), static_cast<int>(K),
                1.0f, 0.0f, &workspace))
                << "ROCm " << dtype_tag << " grouped floating SwiGLU/down failed";
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            TransferEngine::publishCurrentDeviceWrite(grouped_down, stream);

            for (int row = 0; row < M; ++row)
            {
                auto gate_row = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, K});
                auto up_row = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, K});
                std::memcpy(
                    gate_row->mutable_data(),
                    gate->data() + static_cast<size_t>(row) * K,
                    K * sizeof(float));
                std::memcpy(
                    up_row->mutable_data(),
                    up->data() + static_cast<size_t>(row) * K,
                    K * sizeof(float));
                auto serial_down = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{size_t{1}, N});
                ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    gate_row.get(), DeviceId::rocm(rocm_device_id_), stream));
                ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    up_row.get(), DeviceId::rocm(rocm_device_id_), stream));
                ASSERT_NO_THROW(TransferEngine::prepareDeviceInput(
                    serial_down.get(), DeviceId::rocm(rocm_device_id_), stream));

                ASSERT_TRUE(down_kernel.multiply_tensor_with_fused_swiglu(
                    gate_row.get(), up_row.get(), serial_down.get(),
                    1, static_cast<int>(N), static_cast<int>(K),
                    1.0f, 0.0f, &workspace))
                    << "ROCm " << dtype_tag << " serial floating SwiGLU/down failed for row " << row;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                TransferEngine::publishCurrentDeviceWrite(serial_down, stream);

                EXPECT_EQ(
                    std::memcmp(
                        grouped_down->data() + static_cast<size_t>(row) * N,
                        serial_down->data(),
                        N * sizeof(float)),
                    0)
                    << "ROCm " << dtype_tag << " floating SwiGLU/down grouped row must be bitwise serial-decode equivalent";
            }
        }

        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    };

    run_case("fp32", ROCmFloatingPointGemmKernel::Precision::FP32);
    run_case("fp16", ROCmFloatingPointGemmKernel::Precision::FP16);
    run_case("bf16", ROCmFloatingPointGemmKernel::Precision::BF16);

    const auto records =
        PerfStatsCollector::snapshot({"kernel.rocm_floating_grouped_verifier_swiglu_down_calls"});
    uint64_t fp32_calls = 0;
    uint64_t fp16_calls = 0;
    uint64_t bf16_calls = 0;
    for (const auto &record : records)
    {
        if (record.kind != PerfStatRecord::Kind::Counter ||
            record.domain != "kernel" ||
            record.name != "rocm_floating_grouped_verifier_swiglu_down_calls" ||
            record.tags.at("n") != std::to_string(N) ||
            record.tags.at("k") != std::to_string(K))
        {
            continue;
        }
        const std::string &m_tag = record.tags.at("m");
        if (!(m_tag == "2" || m_tag == "3" || m_tag == "4"))
            continue;
        if (record.tags.at("dtype") == "fp32")
            fp32_calls += record.count;
        else if (record.tags.at("dtype") == "fp16")
            fp16_calls += record.count;
        else if (record.tags.at("dtype") == "bf16")
            bf16_calls += record.count;
    }
    EXPECT_EQ(fp32_calls, verifier_rows.size());
    EXPECT_EQ(fp16_calls, verifier_rows.size());
    EXPECT_EQ(bf16_calls, verifier_rows.size());

    PerfStatsCollector::reset();
}

TEST_F(Test__ROCmFloatingPointGemmKernel, GraphCapturedQwen36AlphaBetaM1MatchesReference)
{
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const size_t M = 1, N = 48, K = 5120;

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto output_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
    auto output_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});

    std::mt19937 rng(138);
    std::uniform_real_distribution<float> input_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);
    for (size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = input_dist(rng);
    for (size_t i = 0; i < N * K; ++i)
    {
        weights_alpha->mutable_data()[i] = weight_dist(rng);
        weights_beta->mutable_data()[i] = weight_dist(rng);
    }

    std::vector<float> ref_alpha(M * N);
    std::vector<float> ref_beta(M * N);
    reference_gemm(input->data(), weights_alpha->data(), ref_alpha.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);
    reference_gemm(input->data(), weights_beta->data(), ref_beta.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);

    WorkspaceRequirements reqs;
    reqs.merge(alpha_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    reqs.merge(beta_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(reqs));
    alpha_kernel.bindWorkspace(&workspace);
    beta_kernel.bindWorkspace(&workspace);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    alpha_kernel.setGPUStream(stream);
    beta_kernel.setGPUStream(stream);

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha"},
        {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta"}};

    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemsetAsync(output_alpha->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(output_beta->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
    TransferEngine::publishCurrentDeviceWrite(output_beta, stream);
    const float *actual_alpha = output_alpha->data();
    const float *actual_beta = output_beta->data();
    std::vector<float> got_alpha(actual_alpha, actual_alpha + M * N);
    std::vector<float> got_beta(actual_beta, actual_beta + M * N);

    EXPECT_GT(compute_cosine_similarity(ref_alpha, got_alpha), 0.9999f);
    EXPECT_GT(compute_cosine_similarity(ref_beta, got_beta), 0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    const auto small_n_route = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "kernel" &&
                   record.name == "rocm_fp32_small_n_batched_projection_calls" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "1" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "48" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "5120" &&
                   record.tags.count("batch") != 0 &&
                   record.tags.at("batch") == "2";
        });
    ASSERT_NE(small_n_route, records.end())
        << "Qwen3.6 alpha/beta decode projections should use the ROCm small-N FP32 batched route";

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, GraphCapturedQwen36AlphaBetaPrefillM256MatchesReference)
{
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const size_t M = 256, N = 16, K = 2048;

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto output_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
    auto output_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});

    std::mt19937 rng(140);
    std::uniform_real_distribution<float> input_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);
    for (size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = input_dist(rng);
    for (size_t i = 0; i < N * K; ++i)
    {
        weights_alpha->mutable_data()[i] = weight_dist(rng);
        weights_beta->mutable_data()[i] = weight_dist(rng);
    }

    std::vector<float> ref_alpha(M * N);
    std::vector<float> ref_beta(M * N);
    reference_gemm(input->data(), weights_alpha->data(), ref_alpha.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);
    reference_gemm(input->data(), weights_beta->data(), ref_beta.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);

    WorkspaceRequirements reqs;
    reqs.merge(alpha_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    reqs.merge(beta_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(reqs));
    alpha_kernel.bindWorkspace(&workspace);
    beta_kernel.bindWorkspace(&workspace);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    alpha_kernel.setGPUStream(stream);
    beta_kernel.setGPUStream(stream);

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha"},
        {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta"}};

    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemsetAsync(output_alpha->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(output_beta->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
    TransferEngine::publishCurrentDeviceWrite(output_beta, stream);
    const float *actual_alpha = output_alpha->data();
    const float *actual_beta = output_beta->data();
    std::vector<float> got_alpha(actual_alpha, actual_alpha + M * N);
    std::vector<float> got_beta(actual_beta, actual_beta + M * N);

    EXPECT_GT(compute_cosine_similarity(ref_alpha, got_alpha), 0.9999f);
    EXPECT_GT(compute_cosine_similarity(ref_beta, got_beta), 0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    const auto small_n_route = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "kernel" &&
                   record.name == "rocm_fp32_small_n_batched_projection_calls" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "256" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "16" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "2048" &&
                   record.tags.count("batch") != 0 &&
                   record.tags.at("batch") == "2";
        });
    ASSERT_NE(small_n_route, records.end())
        << "Qwen3.6 prefill alpha/beta projections should use the ROCm small-N FP32 batched route";

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, BatchedFusedProjectionRestagesPointersAfterWorkspaceClobber)
{
    const size_t M = 1, N = 48, K = 5120;

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto output_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
    auto output_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});

    std::mt19937 rng(139);
    std::uniform_real_distribution<float> input_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> weight_dist(-0.25f, 0.25f);
    for (size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = input_dist(rng);
    for (size_t i = 0; i < N * K; ++i)
    {
        weights_alpha->mutable_data()[i] = weight_dist(rng);
        weights_beta->mutable_data()[i] = weight_dist(rng);
    }

    std::vector<float> ref_alpha(M * N);
    std::vector<float> ref_beta(M * N);
    reference_gemm(input->data(), weights_alpha->data(), ref_alpha.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);
    reference_gemm(input->data(), weights_beta->data(), ref_beta.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);

    WorkspaceRequirements reqs;
    reqs.merge(alpha_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    reqs.merge(beta_kernel.getWorkspaceRequirements(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)));
    DeviceWorkspaceManager workspace(DeviceId::rocm(rocm_device_id_), reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(reqs));
    alpha_kernel.bindWorkspace(&workspace);
    beta_kernel.bindWorkspace(&workspace);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    alpha_kernel.setGPUStream(stream);
    beta_kernel.setGPUStream(stream);

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha"},
        {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta"}};

    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    for (const char *name : {
             GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS,
             GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS,
             GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS})
    {
        ASSERT_TRUE(workspace.hasBuffer(name));
        ASSERT_EQ(hipMemsetAsync(workspace.getBuffer(name), 0, workspace.getBufferSize(name), stream), hipSuccess);
    }
    ASSERT_EQ(hipMemsetAsync(output_alpha->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(output_beta->gpu_data_ptr(), 0, M * N * sizeof(float), stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_TRUE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K), nullptr, &workspace));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    TransferEngine::publishCurrentDeviceWrite(output_alpha, stream);
    TransferEngine::publishCurrentDeviceWrite(output_beta, stream);
    const float *actual_alpha = output_alpha->data();
    const float *actual_beta = output_beta->data();
    std::vector<float> got_alpha(actual_alpha, actual_alpha + M * N);
    std::vector<float> got_beta(actual_beta, actual_beta + M * N);

    EXPECT_GT(compute_cosine_similarity(ref_alpha, got_alpha), 0.9999f);
    EXPECT_GT(compute_cosine_similarity(ref_beta, got_beta), 0.9999f);

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, BatchedFusedProjectionRequiresWorkspace)
{
    const size_t M = 2, N = 8, K = 16;

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto weights_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto weights_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto output_alpha = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
    auto output_beta = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    for (size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist(rng);
    for (size_t i = 0; i < N * K; ++i)
    {
        weights_alpha->mutable_data()[i] = dist(rng);
        weights_beta->mutable_data()[i] = dist(rng);
    }

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(weights_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_alpha->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));
    ASSERT_TRUE(output_beta->ensureOnDevice(DeviceId::rocm(rocm_device_id_)));

    ROCmFloatingPointGemmKernel alpha_kernel(weights_alpha.get(), rocm_device_id_);
    ROCmFloatingPointGemmKernel beta_kernel(weights_beta.get(), rocm_device_id_);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    alpha_kernel.setGPUStream(stream);
    beta_kernel.setGPUStream(stream);

    std::vector<ITensorGemm::TensorProjectionDesc> projections = {
        {&alpha_kernel, output_alpha.get(), static_cast<int>(N), nullptr, "alpha"},
        {&beta_kernel, output_beta.get(), static_cast<int>(N), nullptr, "beta"}};

    EXPECT_FALSE(alpha_kernel.multiply_fused_tensor(
        input.get(), projections, static_cast<int>(M), static_cast<int>(K)));
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(Test__ROCmFloatingPointGemmKernel, MappedOutputRedirectRequiresDeclaredWorkspace)
{
    const size_t M = 2, N = 16, K = 32;
    const DeviceId device = DeviceId::rocm(rocm_device_id_);

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    auto weights = std::make_unique<FP32Tensor>(std::vector<size_t>{N, K});
    auto output = FP32Tensor::createMapped(std::vector<size_t>{M, N}, device);
    if (!output || !output->isMapped())
    {
        GTEST_SKIP() << "Mapped ROCm memory allocation not supported on this system";
    }

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    for (size_t i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist(rng);
    for (size_t i = 0; i < N * K; ++i)
        weights->mutable_data()[i] = dist(rng);

    std::vector<float> ref(M * N);
    reference_gemm(input->data(), weights->data(), ref.data(),
                   static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), true);

    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(weights->ensureOnDevice(device));

    ROCmFloatingPointGemmKernel kernel(weights.get(), rocm_device_id_);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    kernel.setGPUStream(stream);

    EXPECT_FALSE(kernel.multiply_tensor(
        input.get(),
        output.get(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        true,
        1.0f,
        0.0f));

    WorkspaceRequirements reqs = kernel.getWorkspaceRequirements(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    reqs.buffers.push_back({
        GemmWorkspaceBuffers::ROCM_FP32_MAPPED_REDIRECT,
        M * N * sizeof(float),
        256,
        true});
    DeviceWorkspaceManager workspace(device, reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace.allocate(reqs));
    kernel.bindWorkspace(&workspace);

    ASSERT_TRUE(kernel.multiply_tensor(
        input.get(),
        output.get(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        true,
        1.0f,
        0.0f,
        nullptr,
        nullptr,
        -1,
        &workspace));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> got(output->data(), output->data() + M * N);
    EXPECT_GT(compute_cosine_similarity(ref, got), 0.9999f);

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

#else // !HAVE_ROCM

// Placeholder test when ROCm is not available
TEST(Test__ROCmFloatingPointGemmKernel, Disabled_NoROCm)
{
    GTEST_SKIP() << "ROCm not compiled in this build";
}

#endif // HAVE_ROCM
