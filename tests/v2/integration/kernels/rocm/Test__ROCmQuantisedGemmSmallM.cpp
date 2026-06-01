/**
 * @file Test__ROCmQuantisedGemmSmallM.cpp
 * @brief Focused ROCm small-M GEMM regressions for MTP verifier decode.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"

#include <cmath>
#include <memory>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
    enum class PackedPath
    {
        INT8VNNI,
        NativeVNNI,
    };

    bool hasROCmDevice()
    {
#ifdef HAVE_ROCM
        int count = 0;
        const hipError_t err = hipGetDeviceCount(&count);
        return err == hipSuccess && count > 0;
#else
        return false;
#endif
    }

    void cpuFP32GemmRef(const float *A, const float *W, float *C, int M, int N, int K)
    {
        for (int row = 0; row < M; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                double acc = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    acc += static_cast<double>(A[row * K + kk]) *
                           static_cast<double>(W[col * K + kk]);
                C[row * N + col] = static_cast<float>(acc);
            }
        }
    }

    float cosineSim(const float *a, const float *b, size_t n)
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (na == 0.0 || nb == 0.0)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    std::unique_ptr<DeviceWorkspaceManager> bindWorkspace(
        ROCmQuantisedGemmKernel &kernel,
        int M,
        int N,
        int K)
    {
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            64 * 1024 * 1024);
        if (!workspace->allocate(kernel.getWorkspaceRequirements(M, N, K)))
            return nullptr;
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

    void expectPackedPath(const ROCmPackedWeights &packed, PackedPath path)
    {
        switch (path)
        {
        case PackedPath::INT8VNNI:
            EXPECT_FALSE(packed.int8_data_vnni.empty());
            EXPECT_TRUE(packed.native_vnni_payload.empty());
            break;
        case PackedPath::NativeVNNI:
            EXPECT_FALSE(packed.native_vnni_payload.empty());
            EXPECT_FALSE(packed.native_vnni_scales.empty());
            EXPECT_TRUE(packed.int8_data_vnni.empty());
            break;
        }
    }

    template <typename CreateWeights>
    void runDispatchM2MatchesReference(
        const char *label,
        int N,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine)
    {
        const int M = 2;

        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            42);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, expected_path);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(input->data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " M=2 cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

        kernel.unbindWorkspace();
    }

    template <typename CreateWeights>
    void runFusedQKVM2MatchesSeparate(
        const char *label,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine)
    {
        const int M = 2;
        const int Nq = 896;
        const int Nk = 128;
        const int Nv = 128;

        auto wq = createWeights({static_cast<size_t>(Nq), static_cast<size_t>(K)}, 101);
        auto wk = createWeights({static_cast<size_t>(Nk), static_cast<size_t>(K)}, 102);
        auto wv = createWeights({static_cast<size_t>(Nv), static_cast<size_t>(K)}, 103);

        ROCmPackedWeights packed_q;
        ROCmPackedWeights packed_k;
        ROCmPackedWeights packed_v;
        ASSERT_TRUE(packWeightsToROCm(wq.get(), packed_q));
        ASSERT_TRUE(packWeightsToROCm(wk.get(), packed_k));
        ASSERT_TRUE(packWeightsToROCm(wv.get(), packed_v));
        expectPackedPath(packed_q, expected_path);
        expectPackedPath(packed_k, expected_path);
        expectPackedPath(packed_v, expected_path);

        ROCmQuantisedGemmKernel q_kernel(&packed_q, 0);
        ROCmQuantisedGemmKernel k_kernel(&packed_k, 0);
        ROCmQuantisedGemmKernel v_kernel(&packed_v, 0);

        auto q_workspace = bindWorkspace(q_kernel, M, Nq, K);
        auto k_workspace = bindWorkspace(k_kernel, M, Nk, K);
        auto v_workspace = bindWorkspace(v_kernel, M, Nv, K);
        ASSERT_NE(q_workspace, nullptr);
        ASSERT_NE(k_workspace, nullptr);
        ASSERT_NE(v_workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto separate_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto separate_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto separate_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto fused_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto fused_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto fused_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto bias_q = TestTensorFactory::createFP32({static_cast<size_t>(Nq)});
        auto bias_k = TestTensorFactory::createFP32({static_cast<size_t>(Nk)});
        auto bias_v = TestTensorFactory::createFP32({static_cast<size_t>(Nv)});

        for (int col = 0; col < Nq; ++col)
            bias_q->mutable_data()[col] = 0.03125f * static_cast<float>((col % 9) - 4);
        for (int col = 0; col < Nk; ++col)
        {
            bias_k->mutable_data()[col] = 0.03125f * static_cast<float>((col % 7) - 3);
            bias_v->mutable_data()[col] = 0.03125f * static_cast<float>((col % 5) - 2);
        }

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(separate_q->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(separate_k->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(separate_v->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_q->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_k->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_v->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(q_kernel.multiply_tensor(input.get(), separate_q.get(), M, Nq, K,
                                             true, 1.0f, 0.0f, bias_q.get()));
        ASSERT_TRUE(k_kernel.multiply_tensor(input.get(), separate_k.get(), M, Nk, K,
                                             true, 1.0f, 0.0f, bias_k.get()));
        ASSERT_TRUE(v_kernel.multiply_tensor(input.get(), separate_v.get(), M, Nv, K,
                                             true, 1.0f, 0.0f, bias_v.get()));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        separate_q->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        separate_k->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        separate_v->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.emplace_back(&q_kernel, fused_q.get(), Nq, bias_q.get(), "q_small_m");
        projections.emplace_back(&k_kernel, fused_k.get(), Nk, bias_k.get(), "k_small_m");
        projections.emplace_back(&v_kernel, fused_v.get(), Nv, bias_v.get(), "v_small_m");

        ASSERT_TRUE(q_kernel.multiply_fused_tensor(input.get(), projections, M, K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        fused_q->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        fused_k->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        fused_v->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        const float q_cos = cosineSim(fused_q->data(), separate_q->data(), static_cast<size_t>(M) * Nq);
        const float k_cos = cosineSim(fused_k->data(), separate_k->data(), static_cast<size_t>(M) * Nk);
        const float v_cos = cosineSim(fused_v->data(), separate_v->data(), static_cast<size_t>(M) * Nv);

        LOG_INFO("[SmallM] " << label << " fused QKV M=2 cosine q="
                             << q_cos << " k=" << k_cos << " v=" << v_cos);
        EXPECT_GT(q_cos, min_cosine);
        EXPECT_GT(k_cos, min_cosine);
        EXPECT_GT(v_cos, min_cosine);

        q_kernel.unbindWorkspace();
        k_kernel.unbindWorkspace();
        v_kernel.unbindWorkspace();
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ80M2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 896;
    const int K = 896;
    runDispatchM2MatchesReference(
        "Q8_0 INT8-VNNI",
        N,
        K,
        PackedPath::INT8VNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KM2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 896;
    const int K = 1024;
    runDispatchM2MatchesReference(
        "Q4_K native-VNNI",
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ80QKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 896;
    runFusedQKVM2MatchesSeparate(
        "Q8_0 INT8-VNNI",
        K,
        PackedPath::INT8VNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); },
        0.9999f);
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 1024;
    runFusedQKVM2MatchesSeparate(
        "Q4_K native-VNNI",
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f);
}
