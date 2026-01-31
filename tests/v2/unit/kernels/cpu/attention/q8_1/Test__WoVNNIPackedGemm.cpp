#include <gtest/gtest.h>

#include "kernels/KernelFactory.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "tensors/Tensors.h"

extern "C" void llaminar2_wo_q8_1_vnni_packed_gemm(
    const void *wo_packed,
    const float *A,
    float *C,
    int m,
    int n,
    int k);

namespace
{
    static float dot(const float *a, const float *b, size_t n)
    {
        float s = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            s += a[i] * b[i];
        }
        return s;
    }

    static float l2_norm(const float *a, size_t n)
    {
        return std::sqrt(std::max(0.0f, dot(a, a, n)));
    }

    static float cosine_similarity(const float *a, const float *b, size_t n)
    {
        const float na = l2_norm(a, n);
        const float nb = l2_norm(b, n);
        if (na == 0.0f || nb == 0.0f)
        {
            return 0.0f;
        }
        return dot(a, b, n) / (na * nb);
    }
}

TEST(Test__WoVNNIPackedGemm, OutputCloseToFP32Reference_64x64)
{
    constexpr int K = 64;
    constexpr int N = 64;
    constexpr int M = 4;

    // Deterministic pseudo-random values (no RNG dependency)
    std::vector<float> wo_fp32(N * K);
    for (int i = 0; i < N * K; ++i)
    {
        wo_fp32[i] = (static_cast<float>((i * 1315423911u) & 1023u) / 1023.0f) * 2.0f - 1.0f;
    }

    std::vector<float> A(M * K);
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = (static_cast<float>((i * 2654435761u) & 1023u) / 1023.0f) * 2.0f - 1.0f;
    }

    auto wo_q8 = llaminar2::Q8_1Tensor::quantize_from_fp32(wo_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    const auto *packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(wo_q8.get());

    std::vector<float> C_vnni(M * N, 0.0f);
    llaminar2_wo_q8_1_vnni_packed_gemm(packed, A.data(), C_vnni.data(), M, N, K);

    // FP32 reference: C_ref[m,n] = sum_k A[m,k] * Wo[n,k]
    std::vector<float> C_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                acc += A[m * K + k] * wo_fp32[n * K + k];
            }
            C_ref[m * N + n] = acc;
        }
    }

    const float cos = cosine_similarity(C_ref.data(), C_vnni.data(), C_ref.size());

    float max_abs = 0.0f;
    for (size_t i = 0; i < C_ref.size(); ++i)
    {
        max_abs = std::max(max_abs, std::abs(C_ref[i] - C_vnni[i]));
    }

    // Quantized activations introduce approximation; keep thresholds conservative.
    EXPECT_GT(cos, 0.98f);
    EXPECT_LT(max_abs, 0.75f);
}
