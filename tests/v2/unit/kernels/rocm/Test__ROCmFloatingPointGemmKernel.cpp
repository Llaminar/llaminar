/**
 * @file Test__ROCmFloatingPointGemmKernel.cpp
 * @brief Unit tests for ROCm floating-point GEMM kernel using hipBLAS
 *
 * Tests the ROCmFloatingPointGemmKernel which wraps hipBLAS for FP32/FP16/BF16
 * GEMM operations on AMD GPUs (MI50, MI100, MI250, etc.)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

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
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

using namespace llaminar2;
using namespace llaminar2::rocm;

namespace
{
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
        hipSetDevice(rocm_device_id_);

        // Get device properties
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, rocm_device_id_);
        LOG_INFO("[Test] Using ROCm device: " << props.name << " (gfx" << props.gcnArchName << ")");
    }

    void TearDown() override
    {
        hipDeviceSynchronize();
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
        hipMalloc(&d_ptr, host_data.size() * sizeof(float));
        hipMemcpy(d_ptr, host_data.data(), host_data.size() * sizeof(float), hipMemcpyHostToDevice);
        return d_ptr;
    }

    void copy_from_gpu(float *d_ptr, std::vector<float> &host_data)
    {
        hipMemcpy(host_data.data(), d_ptr, host_data.size() * sizeof(float), hipMemcpyDeviceToHost);
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
};

// ============================================================================
// HipBLASGemmKernel Tests (Low-level)
// ============================================================================

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
    ASSERT_TRUE(kernel.execute(d_A, d_B, d_C, M, N, K, false, true));

    copy_from_gpu(d_C, h_C);

    // Validate
    float max_rel_err = compute_relative_error(h_ref, h_C);
    float cosine_sim = compute_cosine_similarity(h_ref, h_C);
    LOG_INFO("[Test] Small matrix - max relative error: " << max_rel_err << ", cosine similarity: " << cosine_sim);
    EXPECT_LT(max_rel_err, 1e-5f);
    EXPECT_GT(cosine_sim, 0.9999f); // Expect near-perfect alignment

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
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
    ASSERT_TRUE(kernel.execute(d_A, d_B, d_C, M, N, K, false, true));

    copy_from_gpu(d_C, h_C);

    float max_rel_err = compute_relative_error(h_ref, h_C);
    float cosine_sim = compute_cosine_similarity(h_ref, h_C);
    LOG_INFO("[Test] Qwen 0.5B sizes - max relative error: " << max_rel_err << ", cosine similarity: " << cosine_sim);
    // Large matrix GEMM accumulates rounding errors - 10% tolerance is reasonable
    EXPECT_LT(max_rel_err, 0.1f);
    EXPECT_GT(cosine_sim, 0.999f); // Expect high alignment for GEMM

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
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
    ASSERT_TRUE(kernel.execute(d_A, d_B, d_C, M, N, K, false, true));

    copy_from_gpu(d_C, h_C);

    float max_rel_err = compute_relative_error(h_ref, h_C);
    float cosine_sim = compute_cosine_similarity(h_ref, h_C);
    LOG_INFO("[Test] Qwen 14B sizes - max relative error: " << max_rel_err << ", cosine similarity: " << cosine_sim);
    // Large matrix GEMM accumulates rounding errors - 10% tolerance is reasonable
    EXPECT_LT(max_rel_err, 0.1f);
    EXPECT_GT(cosine_sim, 0.999f); // Expect high alignment for GEMM

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
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

    // Warmup
    kernel.execute(d_A, d_B, d_C, M, N, K, false, true);
    hipDeviceSynchronize();

    // Benchmark
    const int num_iters = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; ++i)
    {
        kernel.execute(d_A, d_B, d_C, M, N, K, false, true);
    }
    hipDeviceSynchronize();

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

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
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
    output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE); // Ensure sync back from GPU
    const float *out_data = output->data();

    float sum = 0.0f;
    for (size_t i = 0; i < M * N; ++i)
        sum += std::abs(out_data[i]);

    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";
    LOG_INFO("[Test] TensorInterface basic test passed, output sum=" << sum);
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

    output_alpha->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_beta->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *actual_alpha = output_alpha->data();
    const float *actual_beta = output_beta->data();
    std::vector<float> got_alpha(actual_alpha, actual_alpha + M * N);
    std::vector<float> got_beta(actual_beta, actual_beta + M * N);

    const float alpha_cosine = compute_cosine_similarity(ref_alpha, got_alpha);
    const float beta_cosine = compute_cosine_similarity(ref_beta, got_beta);
    EXPECT_GT(alpha_cosine, 0.9999f);
    EXPECT_GT(beta_cosine, 0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    const auto tiny_route = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "kernel" &&
                   record.name == "rocm_fp32_tiny_batched_projection_calls" &&
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
    ASSERT_NE(tiny_route, records.end())
        << "Graph-captured M=2 alpha/beta projections should use the ROCm tiny FP32 batched route";
    EXPECT_GE(tiny_route->value, 1.0);

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
        << "Tiny graph-captured FP32 projection shapes should not pay hipBLAS batched launch overhead";

    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
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

    output_alpha->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_beta->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *actual_alpha = output_alpha->data();
    const float *actual_beta = output_beta->data();
    std::vector<float> got_alpha(actual_alpha, actual_alpha + M * N);
    std::vector<float> got_beta(actual_beta, actual_beta + M * N);

    EXPECT_GT(compute_cosine_similarity(ref_alpha, got_alpha), 0.9999f);
    EXPECT_GT(compute_cosine_similarity(ref_beta, got_beta), 0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    const auto tiny_route = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.kind == PerfStatRecord::Kind::Counter &&
                   record.domain == "kernel" &&
                   record.name == "rocm_fp32_tiny_batched_projection_calls" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "1" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "48" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "5120" &&
                   record.tags.count("batch") != 0 &&
                   record.tags.at("batch") == "2";
        });
    ASSERT_NE(tiny_route, records.end())
        << "Qwen3.6 alpha/beta decode projections should use the ROCm tiny FP32 batched route";

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

    output_alpha->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    output_beta->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
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

    WorkspaceRequirements reqs;
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
