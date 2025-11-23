#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <random>

#ifdef HAVE_ONEDNN

#include "oneapi/dnnl/dnnl.hpp"
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "tensors/IQQuantTables.h"
#include "tensors/FP16Utils.h"

using llaminar2::FP32Tensor;
using llaminar2::gemm_v4::OneDNNGemmKernel;

namespace
{
    using dnnl::memory;

    void apply_rowwise_softmax(float *data, int rows, int cols)
    {
        for (int r = 0; r < rows; ++r)
        {
            const size_t base = static_cast<size_t>(r) * static_cast<size_t>(cols);
            float max_val = data[base];
            for (int c = 1; c < cols; ++c)
            {
                max_val = std::max(max_val, data[base + static_cast<size_t>(c)]);
            }

            float sum = 0.0f;
            for (int c = 0; c < cols; ++c)
            {
                float value = std::exp(data[base + static_cast<size_t>(c)] - max_val);
                data[base + static_cast<size_t>(c)] = value;
                sum += value;
            }

            const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int c = 0; c < cols; ++c)
            {
                data[base + static_cast<size_t>(c)] *= inv_sum;
            }
        }
    }

    void apply_columnwise_softmax(float *data, int rows, int cols)
    {
        for (int c = 0; c < cols; ++c)
        {
            float max_val = data[c];
            for (int r = 1; r < rows; ++r)
            {
                max_val = std::max(max_val, data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)]);
            }

            float sum = 0.0f;
            for (int r = 0; r < rows; ++r)
            {
                const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                float value = std::exp(data[idx] - max_val);
                data[idx] = value;
                sum += value;
            }

            const float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int r = 0; r < rows; ++r)
            {
                const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                data[idx] *= inv_sum;
            }
        }
    }

    std::vector<float> run_direct_onednn_matmul(
        const std::vector<float> &a,
        const std::vector<float> &b,
        int m,
        int n,
        int k,
        bool fuse_softmax,
        int axis,
        const std::vector<float> *bias = nullptr,
        float beta = 0.0f,
        const std::vector<float> *initial = nullptr)
    {
        using namespace dnnl;

        engine eng(engine::kind::cpu, 0);
        stream strm(eng);

        memory::dims src_dims = {m, k};
        memory::dims weight_dims = {k, n};
        memory::dims dst_dims = {m, n};

        memory::desc src_md(src_dims, memory::data_type::f32, memory::format_tag::ab);
        memory::desc weight_md(weight_dims, memory::data_type::f32, memory::format_tag::ab);
        memory::desc dst_md(dst_dims, memory::data_type::f32, memory::format_tag::ab);

        primitive_attr attr;
        post_ops ops;
        if (beta != 0.0f)
        {
            ops.append_sum(beta);
        }
        if (fuse_softmax)
        {
            ops.append_softmax(axis);
        }
        if (ops.len() > 0)
        {
            attr.set_post_ops(ops);
        }

        std::vector<float> src_buffer = a;
        std::vector<float> weight_buffer = b;
        std::vector<float> dst_buffer(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
        if (beta != 0.0f)
        {
            if (!initial || initial->size() != dst_buffer.size())
            {
                throw std::invalid_argument("Initial buffer required when beta != 0");
            }
            dst_buffer = *initial;
        }

        memory src_mem(src_md, eng, src_buffer.data());
        memory weight_mem(weight_md, eng, weight_buffer.data());
        memory dst_mem(dst_md, eng, dst_buffer.data());

        std::unique_ptr<memory> bias_mem;
        matmul::primitive_desc matmul_pd = [&]()
        {
            if (bias && !bias->empty())
            {
                memory::dims bias_dims = {1, n};
                memory::desc bias_md(bias_dims, memory::data_type::f32, memory::format_tag::ab);
                bias_mem = std::make_unique<memory>(bias_md, eng, const_cast<float *>(bias->data()));
                return matmul::primitive_desc(eng, src_md, weight_md, bias_md, dst_md, attr);
            }

            return matmul::primitive_desc(eng, src_md, weight_md, dst_md, attr);
        }();

        matmul matmul_primitive(matmul_pd);
        if (bias_mem)
        {
            matmul_primitive.execute(
                strm,
                {{DNNL_ARG_SRC, src_mem},
                 {DNNL_ARG_WEIGHTS, weight_mem},
                 {DNNL_ARG_BIAS, *bias_mem},
                 {DNNL_ARG_DST, dst_mem}});
        }
        else
        {
            matmul_primitive.execute(
                strm,
                {{DNNL_ARG_SRC, src_mem},
                 {DNNL_ARG_WEIGHTS, weight_mem},
                 {DNNL_ARG_DST, dst_mem}});
        }

        strm.wait();
        return dst_buffer;
    }
}

TEST(Test__OneDNNGemmKernel, DirectPrimitiveMatmulMatchesReference)
{
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;

    const std::vector<float> A = {
        0.25f, -0.5f, 1.0f, 0.75f,
        -1.25f, 0.6f, 0.8f, -0.4f};

    const std::vector<float> B = {
        0.5f, -0.25f, 0.9f,
        -0.1f, 0.3f, -0.7f,
        0.0f, 0.2f, 0.4f,
        0.8f, -0.6f, 0.1f};

    auto result = run_direct_onednn_matmul(A, B, m, n, k, false, 1);

    std::vector<float> reference(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            for (int kk = 0; kk < k; ++kk)
            {
                reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)] +=
                    A[static_cast<size_t>(row) * k + static_cast<size_t>(kk)] *
                    B[static_cast<size_t>(kk) * n + static_cast<size_t>(col)];
            }
        }
    }

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(result[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, DirectPrimitiveSoftmaxMatchesReference)
{
    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int k = 3;

    const std::vector<float> A = {
        0.2f, -0.3f, 0.5f,
        0.7f, 0.1f, -0.2f,
        -0.4f, 0.9f, 0.6f};

    const std::vector<float> B = {
        -0.1f, 0.6f,
        0.8f, -0.5f,
        0.3f, 0.4f};

    auto fused = run_direct_onednn_matmul(A, B, m, n, k, true, 1);

    std::vector<float> reference(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            for (int kk = 0; kk < k; ++kk)
            {
                reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)] +=
                    A[static_cast<size_t>(row) * k + static_cast<size_t>(kk)] *
                    B[static_cast<size_t>(kk) * n + static_cast<size_t>(col)];
            }
        }
    }

    apply_rowwise_softmax(reference.data(), m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, DirectPrimitiveLargeShapeSoftmaxMatchesReference)
{
    constexpr int m = 8;
    constexpr int n = 10;
    constexpr int k = 16;

    std::vector<float> A(static_cast<size_t>(m) * static_cast<size_t>(k));
    std::vector<float> B(static_cast<size_t>(k) * static_cast<size_t>(n));

    for (size_t idx = 0; idx < A.size(); ++idx)
    {
        A[idx] = std::sin(static_cast<float>(idx) * 0.13f) * 0.5f + std::cos(static_cast<float>(idx) * 0.07f) * 0.3f;
    }
    for (size_t idx = 0; idx < B.size(); ++idx)
    {
        B[idx] = std::cos(static_cast<float>(idx) * 0.11f) * 0.4f - std::sin(static_cast<float>(idx) * 0.05f) * 0.2f;
    }

    auto fused = run_direct_onednn_matmul(A, B, m, n, k, true, 1);

    std::vector<float> reference(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            for (int kk = 0; kk < k; ++kk)
            {
                reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)] +=
                    A[static_cast<size_t>(row) * k + static_cast<size_t>(kk)] *
                    B[static_cast<size_t>(kk) * n + static_cast<size_t>(col)];
            }
        }
    }

    apply_rowwise_softmax(reference.data(), m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 5e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, DirectPrimitiveBiasAddMatchesReference)
{
    constexpr int m = 2;
    constexpr int n = 4;
    constexpr int k = 3;

    const std::vector<float> A = {
        0.1f, -0.2f, 0.3f,
        0.4f, 0.5f, -0.6f};

    const std::vector<float> B = {
        0.2f, -0.1f, 0.3f, 0.7f,
        -0.4f, 0.6f, -0.2f, 0.5f,
        0.8f, 0.1f, -0.5f, -0.3f};

    const std::vector<float> bias = {0.05f, -0.1f, 0.2f, -0.15f};

    auto result = run_direct_onednn_matmul(A, B, m, n, k, false, 1, &bias);

    std::vector<float> reference(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            for (int kk = 0; kk < k; ++kk)
            {
                reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)] +=
                    A[static_cast<size_t>(row) * k + static_cast<size_t>(kk)] *
                    B[static_cast<size_t>(kk) * n + static_cast<size_t>(col)];
            }
            reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)] += bias[static_cast<size_t>(col)];
        }
    }

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(result[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, DirectPrimitiveBetaScalingMatchesReference)
{
    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int k = 3;

    const std::vector<float> A = {
        -0.2f, 0.6f, -0.8f,
        0.5f, -0.1f, 0.9f,
        0.7f, 0.3f, -0.4f};

    const std::vector<float> B = {
        0.4f, -0.5f,
        0.2f, 0.3f,
        -0.7f, 0.8f};

    std::vector<float> initial(static_cast<size_t>(m) * static_cast<size_t>(n));
    for (size_t idx = 0; idx < initial.size(); ++idx)
    {
        initial[idx] = std::sin(static_cast<float>(idx) * 0.21f);
    }

    constexpr float beta = 0.35f;

    auto result = run_direct_onednn_matmul(A, B, m, n, k, false, 1, nullptr, beta, &initial);

    std::vector<float> reference(initial);
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            float accum = 0.0f;
            for (int kk = 0; kk < k; ++kk)
            {
                accum += A[static_cast<size_t>(row) * k + static_cast<size_t>(kk)] *
                         B[static_cast<size_t>(kk) * n + static_cast<size_t>(col)];
            }

            reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)] =
                accum + beta * reference[static_cast<size_t>(row) * n + static_cast<size_t>(col)];
        }
    }

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(result[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, FusedMatmulSoftmaxMatchesReference)
{
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;

    const float A[m * k] = {
        0.5f, -1.0f, 0.25f, 1.0f,
        1.5f, 0.0f, -0.5f, 0.75f};

    const float B[n * k] = {
        0.2f, -0.3f, 0.1f, 0.4f,
        -0.6f, 0.5f, 0.0f, 0.25f,
        0.3f, 0.2f, -0.4f, -0.1f};

    float fused_output[m * n] = {0};
    float reference[m * n] = {0};

    OneDNNGemmKernel kernel;

    ASSERT_TRUE(kernel.multiply_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        1,
        nullptr,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply_activations(
        A,
        B,
        reference,
        m,
        n,
        k,
        true,
        1.0f,
        0.0f,
        nullptr,
        -1));

    apply_rowwise_softmax(reference, m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused_output[i], reference[i], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, FusedMatmulSoftmaxHandlesNegativeAxis)
{
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;

    const float A[m * k] = {
        0.5f, -1.0f, 0.25f, 1.0f,
        1.5f, 0.0f, -0.5f, 0.75f};

    const float B[n * k] = {
        0.2f, -0.3f, 0.1f, 0.4f,
        -0.6f, 0.5f, 0.0f, 0.25f,
        0.3f, 0.2f, -0.4f, -0.1f};

    float fused_output[m * n] = {0};
    float reference[m * n] = {0};

    OneDNNGemmKernel kernel;

    ASSERT_TRUE(kernel.multiply_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        1,
        nullptr,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply_activations(
        A,
        B,
        reference,
        m,
        n,
        k,
        true,
        1.0f,
        0.0f,
        nullptr,
        -1));

    apply_rowwise_softmax(reference, m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused_output[i], reference[i], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, ColumnSoftmaxFallbackMatchesReference)
{
    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int k = 4;

    const float A[m * k] = {
        0.5f, -1.0f, 0.25f, 1.0f,
        1.5f, 0.0f, -0.5f, 0.75f,
        -0.25f, 0.6f, 0.9f, -1.2f};

    const float B[n * k] = {
        -0.1f, 0.3f, 0.5f, -0.6f,
        0.2f, -0.4f, 0.8f, 0.1f};

    float fused_output[m * n] = {0};
    float reference[m * n] = {0};

    OneDNNGemmKernel kernel;

    ASSERT_TRUE(kernel.multiply_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        0,
        nullptr,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply_activations(
        A,
        B,
        reference,
        m,
        n,
        k,
        true,
        1.0f,
        0.0f,
        nullptr,
        -1));

    apply_columnwise_softmax(reference, m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused_output[i], reference[i], 1e-5f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, RejectsInvalidSoftmaxAxis)
{
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 1;
    const float A[1] = {0.0f};
    const float B[1] = {0.0f};
    float C[1] = {0.0f};

    OneDNNGemmKernel kernel;
    EXPECT_FALSE(kernel.multiply_with_softmax(
        A,
        B,
        C,
        m,
        n,
        k,
        true,
        2,
        nullptr,
        nullptr,
        -1));
}

TEST(Test__OneDNNGemmKernel, WeightPathRejectsInvalidSoftmaxAxis)
{
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 1;
    const float A[1] = {0.0f};
    float C[1] = {0.0f};

    FP32Tensor weights({static_cast<size_t>(n), static_cast<size_t>(k)});
    weights.mutable_data()[0] = 0.0f;

    OneDNNGemmKernel kernel(&weights);
    EXPECT_FALSE(kernel.multiply_with_softmax(
        A,
        /*B_unused=*/nullptr,
        C,
        m,
        n,
        k,
        true,
        /*softmax_axis=*/2,
        /*mask=*/nullptr,
        /*mpi_ctx=*/nullptr,
        /*device=*/-1));
}

TEST(Test__OneDNNGemmKernel, StridedNonTransposedB)
{
    constexpr int m = 2;
    constexpr int k = 3;
    constexpr int n = 4;

    const float A[m * k] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f};

    // B as k x n with extra stride to exercise ldb > n
    const int ldb = n + 2;
    std::vector<float> B(static_cast<size_t>(k) * static_cast<size_t>(ldb), 0.0f);
    const float B_dense[k * n] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f};

    for (int row = 0; row < k; ++row)
    {
        float *dst = B.data() + static_cast<size_t>(row) * static_cast<size_t>(ldb);
        std::memcpy(dst,
                    B_dense + static_cast<size_t>(row) * static_cast<size_t>(n),
                    sizeof(float) * static_cast<size_t>(n));
    }

    std::vector<float> C(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

    OneDNNGemmKernel kernel;

    const int lda = k;
    const int ldc = n;

    bool ok = kernel.multiply_activations_strided(
        A,
        B.data(),
        C.data(),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        /*transpose_B=*/false,
        /*alpha=*/1.0f,
        /*beta=*/0.0f,
        nullptr,
        -1);

    ASSERT_TRUE(ok);

    const float expected[m * n] = {
        1 * 1 + 2 * 5 + 3 * 9, 1 * 2 + 2 * 6 + 3 * 10,
        1 * 3 + 2 * 7 + 3 * 11, 1 * 4 + 2 * 8 + 3 * 12,
        4 * 1 + 5 * 5 + 6 * 9, 4 * 2 + 5 * 6 + 6 * 10,
        4 * 3 + 5 * 7 + 6 * 11, 4 * 4 + 5 * 8 + 6 * 12};

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(C[static_cast<size_t>(i)], expected[i], 1e-4f) << "Mismatch at idx " << i;
    }
}

TEST(Test__OneDNNGemmKernel, FusedMatmulSoftmaxWithMaskMatchesReference)
{
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;

    const float A[m * k] = {
        0.5f, -1.0f, 0.25f, 1.0f,
        1.5f, 0.0f, -0.5f, 0.75f};

    const float B[n * k] = {
        0.2f, -0.3f, 0.1f, 0.4f,
        -0.6f, 0.5f, 0.0f, 0.25f,
        0.3f, 0.2f, -0.4f, -0.1f};

    const float mask[m * n] = {
        -100.0f, 0.0f, -100.0f,
        0.0f, -100.0f, 0.0f};

    float fused_output[m * n] = {0};
    float reference[m * n] = {0};

    OneDNNGemmKernel kernel;

    ASSERT_TRUE(kernel.multiply_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        1,
        mask,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply_activations(
        A,
        B,
        reference,
        m,
        n,
        k,
        true,
        1.0f,
        0.0f,
        nullptr,
        -1));

    // Apply mask
    for (int i = 0; i < m * n; ++i)
    {
        reference[i] += mask[i];
    }

    apply_rowwise_softmax(reference, m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused_output[i], reference[i], 1e-5f) << "Mismatch at idx " << i;
    }
}

/**
 * @brief Test multiply_typed_activations with INT8 activations
 *
 * Verifies that pre-quantized INT8 activations from FusedRMSNormQuantize
 * can be consumed directly by OneDNNGemmKernel without re-quantization.
 */
TEST(Test__OneDNNGemmKernel, TypedActivationsINT8)
{
    using llaminar2::fp32_to_fp16;
    using llaminar2::IQ4_NLBlock;
    using llaminar2::IQ4_NLTensor;
    using llaminar2::TensorFormat;

    // Small test case: 4 tokens × 32 hidden_dim, weight [64, 32]
    const int m = 4;  // batch size (tokens)
    const int k = 32; // hidden dimension (1 IQ4_NL block)
    const int n = 64; // output dimension (2 IQ4_NL blocks)

    // Create mock IQ4_NL weight tensor [n, k] = [64, 32]
    std::vector<size_t> shape = {static_cast<size_t>(n), static_cast<size_t>(k)};
    size_t blocks_per_row = (k + 31) / 32;            // 1 block per row
    size_t total_blocks = n * blocks_per_row;         // 64 blocks
    std::vector<uint8_t> raw_data(total_blocks * 18); // 64 blocks × 18 bytes

    // Initialize blocks with simple values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);
    std::uniform_int_distribution<uint8_t> index_dist(0, 15);

    for (size_t b = 0; b < total_blocks; ++b)
    {
        IQ4_NLBlock *block = reinterpret_cast<IQ4_NLBlock *>(raw_data.data() + b * 18);
        block->d = fp32_to_fp16(scale_dist(rng));
        for (int i = 0; i < 16; ++i)
        {
            uint8_t low = index_dist(rng);
            uint8_t high = index_dist(rng);
            block->qs[i] = (high << 4) | low;
        }
    }

    auto weight_tensor = std::make_unique<IQ4_NLTensor>(shape, raw_data);
    ASSERT_EQ(weight_tensor->shape()[0], n);
    ASSERT_EQ(weight_tensor->shape()[1], k);

    // Create FP32 activations (simulate FusedRMSNormQuantize input)
    std::vector<float> fp32_activations(m * k);
    for (int i = 0; i < m * k; ++i)
    {
        fp32_activations[i] = static_cast<float>(i % 10) * 0.1f - 0.5f; // Range [-0.5, 0.4]
    }

    // Quantize to INT8 per-row (simulate FusedRMSNormQuantize output)
    std::vector<int8_t> int8_activations(m * k);
    std::vector<float> row_scales(m);

    for (int row = 0; row < m; ++row)
    {
        const float *fp32_row = fp32_activations.data() + row * k;
        int8_t *int8_row = int8_activations.data() + row * k;

        // Find max absolute value for this row
        float max_abs = 0.0f;
        for (int col = 0; col < k; ++col)
        {
            max_abs = std::max(max_abs, std::abs(fp32_row[col]));
        }

        // Compute scale (symmetric quantization)
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        row_scales[row] = scale;

        // Quantize
        for (int col = 0; col < k; ++col)
        {
            const float quantized = fp32_row[col] / scale;
            int8_row[col] = static_cast<int8_t>(std::round(std::clamp(quantized, -127.0f, 127.0f)));
        }
    }

    // Create OneDNN GEMM kernel bound to weight tensor
    OneDNNGemmKernel kernel(weight_tensor.get());

    // Execute GEMM with typed INT8 activations
    std::vector<float> output(m * n);
    ASSERT_TRUE(kernel.multiply_typed_activations(
        int8_activations.data(),
        TensorFormat::INT8,
        row_scales.data(),
        output.data(),
        m, n, k,
        true, // transpose_B
        1.0f, // alpha
        0.0f, // beta
        nullptr,
        -1))
        << "multiply_typed_activations failed for INT8 path";

    // Verify output is non-zero and reasonable
    float sum = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";

    // Compare with FP32 reference path (via multiply_activations)
    std::vector<float> reference_output(m * n);
    ASSERT_TRUE(kernel.multiply_activations(
        fp32_activations.data(),
        nullptr, // Use bound weight tensor
        reference_output.data(),
        m, n, k,
        true,
        1.0f,
        0.0f,
        nullptr,
        -1))
        << "multiply_activations failed for FP32 reference";

    // Allow higher tolerance due to INT8 quantization error
    // FP32→INT8→FP32 round-trip introduces ~1-2% relative error
    for (int i = 0; i < m * n; ++i)
    {
        const float abs_error = std::abs(output[i] - reference_output[i]);
        const float rel_error = std::abs(reference_output[i]) > 1e-5f
                                    ? abs_error / std::abs(reference_output[i])
                                    : 0.0f;

        // INT8 quantization: expect <5% relative error on most elements
        EXPECT_LT(rel_error, 0.05f) << "Relative error too high at idx " << i
                                    << " (INT8: " << output[i] << ", FP32: " << reference_output[i] << ")";
    }
}

#else // HAVE_ONEDNN

TEST(Test__OneDNNGemmKernel, SkippedWithoutOneDNN)
{
    GTEST_SKIP() << "OneDNN backend not available";
}

#endif
