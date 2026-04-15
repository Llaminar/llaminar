/**
 * @file Test__CUDAGemmNonDeterminism.cpp
 * @brief Standalone test to reproduce CUDA GEMM non-determinism
 *
 * Exercises CUDAQuantisedGemmKernel multiply_fused_tensor() in isolation,
 * calling the same kernel with the same input N times and comparing outputs.
 *
 * This reproduces the issue seen in LocalPP_HOST_CUDA_CPU parity tests
 * where FFN_UP shows massive cosine variance (0.70-0.99) across runs.
 *
 * Tests:
 *  - Self-consistency: same kernel, same input → same output across N calls
 *  - Concurrent vs sequential dispatch
 *  - Shared workspace vs separate workspace
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "kernels/KernelFactory.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <numeric>
#include <filesystem>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test;

using KernelDeviceType = llaminar::v2::kernels::DeviceType;
using TensorProjectionDesc = llaminar2::ITensorGemm::TensorProjectionDesc;

namespace
{
    constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    constexpr int NUM_REPETITIONS = 10;

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }

    size_t countDiffs(const float *a, const float *b, size_t count)
    {
        size_t diffs = 0;
        for (size_t i = 0; i < count; ++i)
        {
            if (a[i] != b[i])
                ++diffs;
        }
        return diffs;
    }

    float maxAbsDiff(const float *a, const float *b, size_t count)
    {
        float m = 0.0f;
        for (size_t i = 0; i < count; ++i)
            m = std::max(m, std::abs(a[i] - b[i]));
        return m;
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAGemmNonDeterminism : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};
    std::unique_ptr<DeviceWorkspaceManager> workspace_;

    bool setupSharedWorkspace(
        const std::vector<ITensorGemm *> &kernels,
        int M,
        const std::vector<int> &Ns,
        int K)
    {
        WorkspaceRequirements shared_reqs;
        for (size_t i = 0; i < kernels.size(); ++i)
        {
            auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernels[i]);
            if (!ws)
                continue;
            auto reqs = ws->getWorkspaceRequirements(M, Ns[i], K);
            for (const auto &buf : reqs.buffers)
            {
                auto it = std::find_if(
                    shared_reqs.buffers.begin(), shared_reqs.buffers.end(),
                    [&](const WorkspaceDescriptor &e)
                    { return e.name == buf.name; });
                if (it == shared_reqs.buffers.end())
                {
                    shared_reqs.buffers.push_back(buf);
                }
                else
                {
                    it->size_bytes = std::max(it->size_bytes, buf.size_bytes);
                    it->alignment = std::max(it->alignment, buf.alignment);
                    it->required = it->required || buf.required;
                }
            }
        }

        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024);
        if (!workspace_->allocate(shared_reqs))
            return false;

        for (auto *k : kernels)
        {
            auto *ws = dynamic_cast<IWorkspaceConsumer *>(k);
            if (ws)
                ws->bindWorkspace(workspace_.get());
        }
        return true;
    }

    void cleanupSharedWorkspace(const std::vector<ITensorGemm *> &kernels)
    {
        for (auto *k : kernels)
        {
            auto *ws = dynamic_cast<IWorkspaceConsumer *>(k);
            if (ws && ws->hasWorkspace())
                ws->unbindWorkspace();
        }
        workspace_.reset();
    }
};

// ============================================================================
// Test: Fused Gate/Up GEMM self-consistency (N repeated calls)
//
// This directly mirrors the FFN gate_up_proj stage in Qwen2:
//   M=9, K=896, N_gate=N_up=4864
// with shared workspace, calling multiply_fused_tensor repeatedly.
// ============================================================================

TEST_F(Test__CUDAGemmNonDeterminism, FusedGateUp_SelfConsistency)
{
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(MODEL_PATH)) << "Failed to load model";

    // Load layer 0 gate and up weights (same as parity test)
    auto w_gate_base = loader.loadTensor("blk.0.ffn_gate.weight", DeviceId::cpu());
    auto w_up_base = loader.loadTensor("blk.0.ffn_up.weight", DeviceId::cpu());
    ASSERT_NE(w_gate_base, nullptr);
    ASSERT_NE(w_up_base, nullptr);

    auto *w_gate = dynamic_cast<Q4_0Tensor *>(w_gate_base.get());
    auto *w_up = dynamic_cast<Q4_0Tensor *>(w_up_base.get());
    ASSERT_NE(w_gate, nullptr);
    ASSERT_NE(w_up, nullptr);

    const int M = 9; // Qwen2 parity prompt: 9 tokens
    const int N_gate = static_cast<int>(w_gate->shape()[0]); // 4864
    const int N_up = static_cast<int>(w_up->shape()[0]);     // 4864
    const int K = static_cast<int>(w_gate->shape()[1]);      // 896

    std::cout << "FusedGateUp non-determinism test: M=" << M
              << " K=" << K << " N_gate=" << N_gate << " N_up=" << N_up
              << " repetitions=" << NUM_REPETITIONS << "\n";

    // Upload weights to GPU
    ASSERT_TRUE(w_gate->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_up->ensureOnDevice(gpu_device_));

    // Create CUDA kernels
    auto kernel_gate = llaminar::v2::kernels::KernelFactory::createGemm(
        w_gate, KernelDeviceType::CUDA);
    auto kernel_up = llaminar::v2::kernels::KernelFactory::createGemm(
        w_up, KernelDeviceType::CUDA);
    ASSERT_NE(kernel_gate, nullptr);
    ASSERT_NE(kernel_up, nullptr);

    // Set up SHARED workspace (mirrors FusedGateUpGEMMStage)
    ASSERT_TRUE(setupSharedWorkspace(
        {kernel_gate.get(), kernel_up.get()},
        M, {N_gate, N_up}, K));

    // Create fixed input (deterministic seed)
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *in_data = input->mutable_data();
    for (int i = 0; i < M * K; ++i)
        in_data[i] = dist_(rng_);

    // Store reference output from first call
    std::vector<float> ref_gate(M * N_gate);
    std::vector<float> ref_up(M * N_up);

    int gate_diffs_total = 0, up_diffs_total = 0;
    double min_gate_cos = 1.0, min_up_cos = 1.0;
    float max_gate_diff = 0.0f, max_up_diff = 0.0f;

    for (int rep = 0; rep < NUM_REPETITIONS; ++rep)
    {
        // Fresh output tensors each iteration
        auto out_gate = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_gate)});
        auto out_up = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_up)});

        // Build projection descriptors
        std::vector<TensorProjectionDesc> projections;
        projections.emplace_back(kernel_gate.get(), out_gate.get(), N_gate,
                                 nullptr, "gate");
        projections.emplace_back(kernel_up.get(), out_up.get(), N_up,
                                 nullptr, "up");

        // Call fused method with coherence wrapper
        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {out_gate.get(), out_up.get()},
            [&]
            {
                return kernel_gate->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr);
            }));

        const float *gate_data = out_gate->data();
        const float *up_data = out_up->data();

        if (rep == 0)
        {
            // Store reference
            std::memcpy(ref_gate.data(), gate_data, ref_gate.size() * sizeof(float));
            std::memcpy(ref_up.data(), up_data, ref_up.size() * sizeof(float));
            std::cout << "  Rep 0: reference captured\n";
        }
        else
        {
            // Compare against reference
            size_t g_diffs = countDiffs(gate_data, ref_gate.data(), ref_gate.size());
            size_t u_diffs = countDiffs(up_data, ref_up.data(), ref_up.size());
            double g_cos = cosineSimilarity(gate_data, ref_gate.data(), ref_gate.size());
            double u_cos = cosineSimilarity(up_data, ref_up.data(), ref_up.size());
            float g_max = maxAbsDiff(gate_data, ref_gate.data(), ref_gate.size());
            float u_max = maxAbsDiff(up_data, ref_up.data(), ref_up.size());

            gate_diffs_total += (g_diffs > 0 ? 1 : 0);
            up_diffs_total += (u_diffs > 0 ? 1 : 0);
            min_gate_cos = std::min(min_gate_cos, g_cos);
            min_up_cos = std::min(min_up_cos, u_cos);
            max_gate_diff = std::max(max_gate_diff, g_max);
            max_up_diff = std::max(max_up_diff, u_max);

            std::cout << "  Rep " << rep << ": gate diffs=" << g_diffs
                      << "/" << ref_gate.size()
                      << " cos=" << std::fixed << std::setprecision(6) << g_cos
                      << " max_abs=" << std::scientific << g_max
                      << " | up diffs=" << u_diffs
                      << "/" << ref_up.size()
                      << " cos=" << std::fixed << std::setprecision(6) << u_cos
                      << " max_abs=" << std::scientific << u_max << "\n";
        }
    }

    std::cout << "\n=== SUMMARY ===\n"
              << "  Gate: " << gate_diffs_total << "/" << (NUM_REPETITIONS - 1)
              << " runs had diffs, min_cos=" << std::fixed << std::setprecision(6) << min_gate_cos
              << " max_abs=" << std::scientific << max_gate_diff << "\n"
              << "  Up:   " << up_diffs_total << "/" << (NUM_REPETITIONS - 1)
              << " runs had diffs, min_cos=" << std::fixed << std::setprecision(6) << min_up_cos
              << " max_abs=" << std::scientific << max_up_diff << "\n";

    // The test passes if cosine is very high (>= 0.9999) across all repetitions.
    // If this FAILS, it means the kernel is non-deterministic.
    EXPECT_GE(min_gate_cos, 0.9999)
        << "Gate projection is non-deterministic across " << NUM_REPETITIONS << " calls";
    EXPECT_GE(min_up_cos, 0.9999)
        << "Up projection is non-deterministic across " << NUM_REPETITIONS << " calls";

    cleanupSharedWorkspace({kernel_gate.get(), kernel_up.get()});
}

// ============================================================================
// Test: Fused QKV GEMM self-consistency (N repeated calls)
//
// Mirrors the attention QKV stage: M=9, K=896, N_q=896, N_k=128, N_v=128
// ============================================================================

TEST_F(Test__CUDAGemmNonDeterminism, FusedQKV_SelfConsistency)
{
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(MODEL_PATH)) << "Failed to load model";

    auto w_q_base = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto w_k_base = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto w_v_base = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(w_q_base, nullptr);
    ASSERT_NE(w_k_base, nullptr);
    ASSERT_NE(w_v_base, nullptr);

    auto *w_q = dynamic_cast<Q4_0Tensor *>(w_q_base.get());
    auto *w_k = dynamic_cast<Q4_0Tensor *>(w_k_base.get());
    auto *w_v = dynamic_cast<Q4_0Tensor *>(w_v_base.get());
    ASSERT_NE(w_q, nullptr);
    ASSERT_NE(w_k, nullptr);
    ASSERT_NE(w_v, nullptr);

    const int M = 9;
    const int N_q = static_cast<int>(w_q->shape()[0]);
    const int N_k = static_cast<int>(w_k->shape()[0]);
    const int N_v = static_cast<int>(w_v->shape()[0]);
    const int K = static_cast<int>(w_q->shape()[1]);

    std::cout << "FusedQKV non-determinism test: M=" << M
              << " K=" << K << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v
              << " repetitions=" << NUM_REPETITIONS << "\n";

    ASSERT_TRUE(w_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_v->ensureOnDevice(gpu_device_));

    auto k_q = llaminar::v2::kernels::KernelFactory::createGemm(w_q, KernelDeviceType::CUDA);
    auto k_k = llaminar::v2::kernels::KernelFactory::createGemm(w_k, KernelDeviceType::CUDA);
    auto k_v = llaminar::v2::kernels::KernelFactory::createGemm(w_v, KernelDeviceType::CUDA);
    ASSERT_NE(k_q, nullptr);
    ASSERT_NE(k_k, nullptr);
    ASSERT_NE(k_v, nullptr);

    ASSERT_TRUE(setupSharedWorkspace(
        {k_q.get(), k_k.get(), k_v.get()},
        M, {N_q, N_k, N_v}, K));

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref_q(M * N_q), ref_k(M * N_k), ref_v(M * N_v);
    double min_q_cos = 1.0, min_k_cos = 1.0, min_v_cos = 1.0;

    for (int rep = 0; rep < NUM_REPETITIONS; ++rep)
    {
        auto out_q = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
        auto out_k = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
        auto out_v = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

        std::vector<TensorProjectionDesc> projections;
        projections.emplace_back(k_q.get(), out_q.get(), N_q, nullptr, "Q");
        projections.emplace_back(k_k.get(), out_k.get(), N_k, nullptr, "K");
        projections.emplace_back(k_v.get(), out_v.get(), N_v, nullptr, "V");

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {out_q.get(), out_k.get(), out_v.get()},
            [&]
            {
                return k_q->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr);
            }));

        const float *q_data = out_q->data();
        const float *k_data = out_k->data();
        const float *v_data = out_v->data();

        if (rep == 0)
        {
            std::memcpy(ref_q.data(), q_data, ref_q.size() * sizeof(float));
            std::memcpy(ref_k.data(), k_data, ref_k.size() * sizeof(float));
            std::memcpy(ref_v.data(), v_data, ref_v.size() * sizeof(float));
            std::cout << "  Rep 0: reference captured\n";
        }
        else
        {
            double q_cos = cosineSimilarity(q_data, ref_q.data(), ref_q.size());
            double k_cos = cosineSimilarity(k_data, ref_k.data(), ref_k.size());
            double v_cos = cosineSimilarity(v_data, ref_v.data(), ref_v.size());
            size_t q_diffs = countDiffs(q_data, ref_q.data(), ref_q.size());
            size_t k_diffs = countDiffs(k_data, ref_k.data(), ref_k.size());
            size_t v_diffs = countDiffs(v_data, ref_v.data(), ref_v.size());

            min_q_cos = std::min(min_q_cos, q_cos);
            min_k_cos = std::min(min_k_cos, k_cos);
            min_v_cos = std::min(min_v_cos, v_cos);

            std::cout << "  Rep " << rep
                      << ": Q diffs=" << q_diffs << " cos=" << std::fixed << std::setprecision(6) << q_cos
                      << " | K diffs=" << k_diffs << " cos=" << k_cos
                      << " | V diffs=" << v_diffs << " cos=" << v_cos << "\n";
        }
    }

    std::cout << "\n=== SUMMARY ===\n"
              << "  Q: min_cos=" << std::fixed << std::setprecision(6) << min_q_cos << "\n"
              << "  K: min_cos=" << min_k_cos << "\n"
              << "  V: min_cos=" << min_v_cos << "\n";

    EXPECT_GE(min_q_cos, 0.9999) << "Q projection non-deterministic";
    EXPECT_GE(min_k_cos, 0.9999) << "K projection non-deterministic";
    EXPECT_GE(min_v_cos, 0.9999) << "V projection non-deterministic";

    cleanupSharedWorkspace({k_q.get(), k_k.get(), k_v.get()});
}

// ============================================================================
// Test: Single multiply_tensor self-consistency (not fused)
//
// Calls multiply_tensor() separately for a single kernel N times.
// If THIS is non-deterministic, it's the inner kernel (split-K atomicAdd).
// If this is deterministic but fused is not, it's the workspace sharing.
// ============================================================================

TEST_F(Test__CUDAGemmNonDeterminism, SingleKernel_SelfConsistency)
{
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(MODEL_PATH)) << "Failed to load model";

    // Use FFN up weight — same tensor that shows variance
    auto w_up_base = loader.loadTensor("blk.0.ffn_up.weight", DeviceId::cpu());
    ASSERT_NE(w_up_base, nullptr);

    auto *w_up = dynamic_cast<Q4_0Tensor *>(w_up_base.get());
    ASSERT_NE(w_up, nullptr);

    const int M = 9;
    const int N = static_cast<int>(w_up->shape()[0]);  // 4864
    const int K = static_cast<int>(w_up->shape()[1]);  // 896

    std::cout << "Single kernel non-determinism test: M=" << M
              << " N=" << N << " K=" << K
              << " repetitions=" << NUM_REPETITIONS << "\n";

    ASSERT_TRUE(w_up->ensureOnDevice(gpu_device_));

    auto kernel = llaminar::v2::kernels::KernelFactory::createGemm(
        w_up, KernelDeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);

    // Set up workspace for single kernel
    auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
    if (ws)
    {
        auto reqs = ws->getWorkspaceRequirements(M, N, K);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024);
        ASSERT_TRUE(workspace_->allocate(reqs));
        ws->bindWorkspace(workspace_.get());
    }

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref(M * N);
    double min_cos = 1.0;
    float worst_max_diff = 0.0f;

    for (int rep = 0; rep < NUM_REPETITIONS; ++rep)
    {
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {output.get()},
            [&]
            {
                return kernel->multiply_tensor(
                    input.get(), output.get(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
            }));

        const float *data = output->data();

        if (rep == 0)
        {
            std::memcpy(ref.data(), data, ref.size() * sizeof(float));
            std::cout << "  Rep 0: reference captured\n";
        }
        else
        {
            size_t diffs = countDiffs(data, ref.data(), ref.size());
            double cos = cosineSimilarity(data, ref.data(), ref.size());
            float md = maxAbsDiff(data, ref.data(), ref.size());
            min_cos = std::min(min_cos, cos);
            worst_max_diff = std::max(worst_max_diff, md);

            std::cout << "  Rep " << rep << ": diffs=" << diffs << "/" << ref.size()
                      << " cos=" << std::fixed << std::setprecision(6) << cos
                      << " max_abs=" << std::scientific << md << "\n";
        }
    }

    std::cout << "\n=== SUMMARY ===\n"
              << "  min_cos=" << std::fixed << std::setprecision(6) << min_cos
              << " worst_max_abs=" << std::scientific << worst_max_diff << "\n";

    // DIAGNOSTIC: This test reveals whether individual kernel calls are deterministic.
    // We expect split-K with atomicAdd to show ULP-level diffs, not massive corruption.
    EXPECT_GE(min_cos, 0.9999)
        << "Single multiply_tensor is non-deterministic across " << NUM_REPETITIONS << " calls";

    if (ws)
    {
        ws->unbindWorkspace();
        workspace_.reset();
    }
}

// ============================================================================
// Custom main with MPI initialization
// ============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
