/**
 * @file Test__CuBLASGemm.cpp
 * @brief Test cuBLAS GEMM kernel correctness
 *
 * **Purpose**: Validate CuBLASGemmKernel against CPU reference implementations.
 *
 * **Tests**:
 * - Small matrix correctness
 * - Various sizes (square, tall-skinny, short-wide)
 * - Transpose variants (NN, NT, TN, TT)
 * - Edge cases (single row for decode, large prefill)
 * - Expert-adapter retirement must not drain unrelated inference streams.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>

// Include project headers BEFORE CUDATestUtils.h (provides include paths)
#include "backends/ComputeBackend.h" // DeviceManager
#include "execution/local_execution/device/DeviceContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/gemm/CuBLASGemmKernel.h"
#include "kernels/cuda/gemm/CUDAFloatingPointGemmKernel.h"
#include <cuda_runtime.h>
#endif

// Now include test utils (uses headers above)
#include "../../../utils/CUDATestUtils.h"

using namespace llaminar2;
using namespace llaminar2::test::cuda;

#ifdef HAVE_CUDA
namespace
{
    /** @brief Test-only nonblocking CUDA stream parked behind a host gate. */
    class HostBlockedCudaStream final
    {
    public:
        /** @brief Create the stream and wait until its gate is actively blocking. */
        HostBlockedCudaStream()
        {
            if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
                cudaSuccess)
            {
                throw std::runtime_error(
                    "Could not create adversarial CUDA stream");
            }
            if (cudaLaunchHostFunc(
                    stream_,
                    [](void *opaque)
                    {
                        auto *self = static_cast<HostBlockedCudaStream *>(opaque);
                        self->entered_.store(true, std::memory_order_release);
                        while (!self->release_.load(std::memory_order_acquire))
                            std::this_thread::yield();
                    },
                    this) != cudaSuccess)
            {
                (void)cudaStreamDestroy(stream_);
                stream_ = nullptr;
                throw std::runtime_error(
                    "Could not enqueue adversarial CUDA host gate");
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
                (void)cudaStreamSynchronize(stream_);
                (void)cudaStreamDestroy(stream_);
                stream_ = nullptr;
                throw std::runtime_error(
                    "Adversarial CUDA host gate did not start");
            }
        }

        /** @brief Release pending work before destroying the owned stream. */
        ~HostBlockedCudaStream()
        {
            release();
            if (stream_)
            {
                (void)cudaStreamDestroy(stream_);
            }
        }

        HostBlockedCudaStream(const HostBlockedCudaStream &) = delete;
        HostBlockedCudaStream &operator=(const HostBlockedCudaStream &) = delete;

        /** @return Exact non-default stream parked behind the host gate. */
        [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

        /** @brief Release the gate and wait for its test-only host work to exit. */
        void release() noexcept
        {
            release_.store(true, std::memory_order_release);
            if (stream_)
                (void)cudaStreamSynchronize(stream_);
        }

    private:
        cudaStream_t stream_ = nullptr;
        std::atomic<bool> entered_{false};
        std::atomic<bool> release_{false};
    };
} // namespace
#endif

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CuBLASGemm : public CUDATestBase
{
protected:
#ifdef HAVE_CUDA
    void SetUp() override
    {
        CUDATestBase::SetUp();

        // Create cuBLAS kernel using detected GPU device
        // Note: CuBLASGemmKernel expects CUDA device index (0-based within CUDA devices)
        // gpu_idx_ is the DeviceManager index which includes CPU at index 0
        // For now, use 0 as the CUDA device since we only have one CUDA GPU
        if (gpu_idx_ >= 0)
        {
            kernel_ = std::make_unique<cuda::CuBLASGemmKernel>(0);
            ASSERT_EQ(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                cudaSuccess);
            kernel_->bindStream(ExplicitGPUStream{stream_});
            const auto requirements = kernel_->getWorkspaceRequirements(1, 1, 1);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::cuda(0), requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspace_->allocate(requirements));
            kernel_->bindWorkspace(workspace_.get());
        }
    }

    void TearDown() override
    {
        if (stream_)
        {
            (void)cudaStreamSynchronize(stream_);
        }
        kernel_.reset();
        workspace_.reset();
        if (stream_)
        {
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        CUDATestBase::TearDown();
    }

    std::unique_ptr<cuda::CuBLASGemmKernel> kernel_;
    std::unique_ptr<DeviceWorkspaceManager> workspace_;
    cudaStream_t stream_ = nullptr;
#endif
};

// ============================================================================
// Basic Correctness Tests
// ============================================================================

#ifdef HAVE_CUDA

TEST_F(Test__CuBLASGemm, SmallMatrix_NN)
{
    // Small test: C[32×64] = A[32×128] @ B[128×64]
    const int M = 32, N = 64, K = 128;

    // Generate test data
    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 100);
    auto B = generateRandomFP32(K * N, -1.0f, 1.0f, 200);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference (no transpose)
    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, K * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    // Upload
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, M * N * sizeof(float)));

    // Execute cuBLAS GEMM (no transpose)
    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K,
                                 /*transA=*/false, /*transB=*/false));

    // Download result
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "cuBLAS GEMM (NN) failed parity check";
    result.print();

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, SmallMatrix_NT)
{
    // Test with B transposed (common for weights)
    // C[32×64] = A[32×128] @ B^T where B is stored as [64×128]
    const int M = 32, N = 64, K = 128;

    auto A = generateRandomFP32(M * K, -1.0f, 1.0f, 101);
    auto B = generateRandomFP32(N * K, -1.0f, 1.0f, 201); // B stored as [N×K]
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    // CPU reference (B transposed - use cpuGemmNT)
    cpuGemmNT(A.data(), B.data(), C_cpu.data(), M, N, K);

    // GPU execution
    float *d_A, *d_B, *d_C;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, M * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, N * K * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, M * N * sizeof(float)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    // Execute with transB=true
    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K,
                                 /*transA=*/false, /*transB=*/true));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "cuBLAS GEMM (NT) failed parity check";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * @brief Prove independent CUDA projections cannot inherit another stream.
 *
 * CUDA floating projections own separate cuBLAS handles.  The second handle is
 * deliberately bound to a blocked stream after the first handle is bound.  The
 * first projection must still complete on its own stream, and clearing its
 * binding must make the next launch fail instead of reusing stale library state.
 */
/**
 * @brief Retiring any floating expert must not destroy device-wide BLAS resources.
 *
 * The weight allocation outlives all adapters. Only projection-object lifetime
 * ends inside the adversarial interval, matching a completed expert transfer
 * releasing its old bank. A host gate represents unrelated unfinished inference;
 * the test releases it before joining so the broken destructor fails boundedly.
 */
TEST_F(Test__CuBLASGemm, FloatingExpertRetirementDoesNotWaitForUnrelatedStream)
{
    using Adapter = cuda::CUDAFloatingPointGemmKernel;
    void *weights = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&weights, 16 * 16 * sizeof(float)));
    std::vector<std::unique_ptr<Adapter>> projections;
    for (const auto precision : {Adapter::Precision::FP16,
                                 Adapter::Precision::BF16,
                                 Adapter::Precision::FP32})
        projections.push_back(std::make_unique<Adapter>(
            weights, 16, 16, 0, precision, std::shared_ptr<void>{}));

    HostBlockedCudaStream blocked;
    std::promise<void> retired;
    auto completion = retired.get_future();
    std::thread retirement([&]
    {
        (void)cudaSetDevice(0);
        projections.clear();
        retired.set_value();
    });
    const auto status = completion.wait_for(std::chrono::seconds(2));
    blocked.release();
    retirement.join();
    EXPECT_EQ(status, std::future_status::ready)
        << "Expert retirement waited for unrelated device work (private BLAS teardown)";
    ASSERT_EQ(cudaSuccess, cudaFree(weights));
}

/** @brief A shared context handle captures independent stream/workspace pairs exactly. */
TEST_F(Test__CuBLASGemm, ContextHandleCapturedConcurrentStreamsReplayExactly)
{
    constexpr int M = 8, N = 64, K = 128;
    const auto a = generateRandomFP32(M * K, -0.2f, 0.2f, 0x5345);
    const auto b = generateRandomFP32(N * K, -0.2f, 0.2f, 0x7210);
    float *d_a = nullptr, *d_b = nullptr, *d_c0 = nullptr, *d_c1 = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_a, a.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_b, b.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_c0, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_c1, M * N * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    cuda::CuBLASGemmKernel secondary(0);
    const auto requirements = secondary.getWorkspaceRequirements(M, N, K);
    DeviceWorkspaceManager side_workspace(DeviceId::cuda(0), requirements.total_bytes_with_alignment());
    ASSERT_TRUE(side_workspace.allocate(requirements));
    secondary.bindWorkspace(&side_workspace);
    secondary.bindStream(ExplicitGPUStream{side});

    // Warm the identical library path once before recording. Save its exact
    // result, then prove replay has not mixed either stream's mutable bindings.
    ASSERT_TRUE(kernel_->execute(d_a, d_b, d_c0, M, N, K, false, false));
    ASSERT_TRUE(secondary.execute(d_a, d_b, d_c1, M, N, K, false, false));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(side));
    std::vector<float> expected(M * N), actual(M * N);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(expected.data(), d_c0, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(fork, stream_));
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(side, fork, 0));
    ASSERT_TRUE(kernel_->execute(d_a, d_b, d_c0, M, N, K, false, false));
    ASSERT_TRUE(secondary.execute(d_a, d_b, d_c1, M, N, K, false, false));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(join, side));
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(stream_, join, 0));
    ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream_, &graph));
    ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 20; ++replay)
    {
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(d_c0, 0x7f, expected.size() * sizeof(float), stream_));
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(d_c1, 0x7f, expected.size() * sizeof(float), stream_));
        ASSERT_EQ(cudaSuccess, cudaGraphLaunch(executable, stream_));
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
        for (auto *output : {d_c0, d_c1})
        {
            ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)))
                << "replay=" << replay;
        }
    }
    ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(executable));
    ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(join));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(fork));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(side));
    ASSERT_EQ(cudaSuccess, cudaFree(d_c1));
    ASSERT_EQ(cudaSuccess, cudaFree(d_c0));
    ASSERT_EQ(cudaSuccess, cudaFree(d_b));
    ASSERT_EQ(cudaSuccess, cudaFree(d_a));
}

TEST_F(Test__CuBLASGemm, ContextHandlesKeepEachProjectionOnItsExactStream)
{
    constexpr int M = 8;
    constexpr int N = 64;
    constexpr int K = 128;

    auto A = generateRandomFP32(M * K, -0.2f, 0.2f, 0x51de);
    auto B = generateRandomFP32(K * N, -0.2f, 0.2f, 0x71de);
    std::vector<float> expected(M * N, 0.0f);
    std::vector<float> actual(M * N, 0.0f);
    cpuGemmNN(A.data(), B.data(), expected.data(), M, N, K);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_observed = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, A.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, B.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, actual.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_observed, actual.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(
        d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(
        d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, actual.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_observed, 0, actual.size() * sizeof(float)));

    cuda::CuBLASGemmKernel projection_a(0);
    cuda::CuBLASGemmKernel projection_b(0);
    projection_a.bindStream(ExplicitGPUStream{stream_});
    projection_a.bindWorkspace(workspace_.get());

    // Pay cuBLAS's one-time algorithm/module initialization before parking an
    // unrelated stream.  The adversarial interval below is intended to test
    // stream ownership, not CUDA's process-wide lazy loader.
    ASSERT_TRUE(projection_a.execute(
        d_A, d_B, d_C, M, N, K, false, false));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(cudaSuccess, cudaMemset(d_C, 0, actual.size() * sizeof(float)));

    HostBlockedCudaStream blocked;
    projection_b.bindStream(ExplicitGPUStream{blocked.get()});

    ASSERT_TRUE(projection_a.execute(
        d_A, d_B, d_C, M, N, K, false, false));

    // Snapshot the result on A's stream before releasing B.  A device-to-device
    // copy stays asynchronous even though the eventual host vector is pageable,
    // so it cannot introduce a hidden device-wide wait on the blocked stream.
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            d_observed,
            d_C,
            actual.size() * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_));

    cudaEvent_t projection_done = nullptr;
    ASSERT_EQ(
        cudaSuccess,
        cudaEventCreateWithFlags(&projection_done, cudaEventDisableTiming));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(projection_done, stream_));
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    cudaError_t completion = cudaErrorNotReady;
    while (completion == cudaErrorNotReady &&
           std::chrono::steady_clock::now() < deadline)
    {
        completion = cudaEventQuery(projection_done);
        std::this_thread::yield();
    }

    // Release the adversarial stream before any potentially global runtime
    // cleanup, even when the bounded completion check is about to fail.
    blocked.release();
    ASSERT_EQ(completion, cudaSuccess)
        << "Projection A did not complete independently of projection B";
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(projection_done));
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            actual.data(),
            d_observed,
            actual.size() * sizeof(float),
            cudaMemcpyDeviceToHost));

    const auto result = compareArrays(
        actual.data(), expected.data(), actual.size(),
        GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed);

    projection_a.clearStreamBinding();
    EXPECT_FALSE(projection_a.execute(
        d_A, d_B, d_C, M, N, K, false, false))
        << "Clearing a CUDA binding must not leave the cuBLAS handle usable";

    (void)cudaFree(d_A);
    (void)cudaFree(d_B);
    (void)cudaFree(d_C);
    (void)cudaFree(d_observed);
}

TEST_F(Test__CuBLASGemm, SquareMatrix_256)
{
    const int M = 256, N = 256, K = 256;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "256×256 GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// LLM-Relevant Size Tests
// ============================================================================

TEST_F(Test__CuBLASGemm, DecodeSize_SingleToken)
{
    // Decode: 1 token through FFN
    // C[1×3584] = A[1×896] @ B[896×3584]  (Qwen2.5-0.5B FFN up)
    const int M = 1, N = 3584, K = 896;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "Single token decode GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, PrefillSize_MediumBatch)
{
    // Prefill: 128 tokens through attention projection
    // C[128×896] = A[128×896] @ B[896×896]
    const int M = 128, N = 896, K = 896;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "Prefill GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, LMHeadSize)
{
    // LM head: project to vocab
    // C[1×151936] = A[1×896] @ B[896×151936]  (Qwen2.5 vocab size)
    // Using smaller vocab for speed
    const int M = 1, N = 32000, K = 896;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "LM head GEMM failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Alpha/Beta Tests
// ============================================================================

TEST_F(Test__CuBLASGemm, AlphaBetaScaling)
{
    const int M = 32, N = 32, K = 64;
    const float alpha = 0.5f;
    const float beta = 0.25f;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    auto C_init = generateRandomFP32(M * N, 0.0f, 1.0f, 300);
    std::vector<float> C_cuda = C_init;
    std::vector<float> C_cpu = C_init;

    // CPU reference with alpha/beta
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p)
            {
                sum += A[i * K + p] * B[p * N + j];
            }
            C_cpu[i * N + j] = alpha * sum + beta * C_cpu[i * N + j];
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_init.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false, alpha, beta));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "Alpha/Beta scaling failed";
    result.print();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__CuBLASGemm, TinyMatrix_4x4)
{
    const int M = 4, N = 4, K = 4;

    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    cpuGemmNN(A.data(), B.data(), C_cpu.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, M, N, K, false, false));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    auto result = compareArrays(C_cuda.data(), C_cpu.data(), M * N, GEMM_ABS_TOL, GEMM_REL_TOL);
    EXPECT_TRUE(result.passed) << "4x4 GEMM failed";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CuBLASGemm, NonSquareAspectRatios)
{
    // Test various aspect ratios

    struct TestCase
    {
        int M, N, K;
        const char *name;
    };

    std::vector<TestCase> cases = {
        {16, 512, 64, "Wide output"},
        {512, 16, 64, "Tall output"},
        {64, 64, 1024, "Deep K"},
        {1024, 64, 64, "Many rows"},
    };

    for (const auto &tc : cases)
    {
        auto A = generateRandomFP32(tc.M * tc.K);
        auto B = generateRandomFP32(tc.K * tc.N);
        std::vector<float> C_cuda(tc.M * tc.N, 0.0f);
        std::vector<float> C_cpu(tc.M * tc.N, 0.0f);

        cpuGemmNN(A.data(), B.data(), C_cpu.data(), tc.M, tc.N, tc.K);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, tc.M * tc.K * sizeof(float));
        cudaMalloc(&d_B, tc.K * tc.N * sizeof(float));
        cudaMalloc(&d_C, tc.M * tc.N * sizeof(float));

        cudaMemcpy(d_A, A.data(), tc.M * tc.K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), tc.K * tc.N * sizeof(float), cudaMemcpyHostToDevice);

        ASSERT_TRUE(kernel_->execute(d_A, d_B, d_C, tc.M, tc.N, tc.K, false, false))
            << "Failed for case: " << tc.name;

        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
        cudaMemcpy(C_cuda.data(), d_C, tc.M * tc.N * sizeof(float), cudaMemcpyDeviceToHost);

        auto result = compareArrays(C_cuda.data(), C_cpu.data(), tc.M * tc.N, GEMM_ABS_TOL, GEMM_REL_TOL);
        EXPECT_TRUE(result.passed) << "Failed for case: " << tc.name;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

#endif // HAVE_CUDA

// ============================================================================
// Non-CUDA Tests (always run)
// ============================================================================

TEST(Test__CuBLASGemm_NoCUDA, SkipsGracefully)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA not available in this build";
#else
    // This test only runs if CUDA is available
    SUCCEED() << "CUDA is available";
#endif
}
