#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef HAVE_ONEDNN

#include "oneapi/dnnl/dnnl.hpp"
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

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
                matmul::desc matmul_desc(src_md, weight_md, bias_md, dst_md);
                return matmul::primitive_desc(matmul_desc, attr, eng);
            }

            matmul::desc matmul_desc(src_md, weight_md, dst_md);
            return matmul::primitive_desc(matmul_desc, attr, eng);
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

TEST(Test__OneDNNGemmKernel, DirectPrimitiveColumnSoftmaxMatchesReference)
{
    constexpr int m = 4;
    constexpr int n = 3;
    constexpr int k = 2;

    const std::vector<float> A = {
        -0.2f, 0.9f,
        0.3f, -0.5f,
        1.1f, -0.7f,
        -0.4f, 0.8f};

    const std::vector<float> B = {
        0.6f, -0.1f, 0.2f,
        -0.3f, 0.4f, -0.8f};

    auto fused = run_direct_onednn_matmul(A, B, m, n, k, true, 0);

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

    apply_columnwise_softmax(reference.data(), m, n);

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

    ASSERT_TRUE(kernel.multiply_activations_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        1,
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

    ASSERT_TRUE(kernel.multiply_activations_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        -1,
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

    ASSERT_TRUE(kernel.multiply_activations_with_softmax(
        A,
        B,
        fused_output,
        m,
        n,
        k,
        true,
        0,
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

TEST(Test__OneDNNGemmKernel, WeightFusedMatmulSoftmaxMatchesReference)
{
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;

    const float A[m * k] = {
        0.5f, -1.0f, 0.25f, 1.0f,
        1.5f, 0.0f, -0.5f, 0.75f};

    const float weight_values[n * k] = {
        0.2f, -0.3f, 0.1f, 0.4f,
        -0.6f, 0.5f, 0.0f, 0.25f,
        0.3f, 0.2f, -0.4f, -0.1f};

    FP32Tensor weights({static_cast<size_t>(n), static_cast<size_t>(k)});
    std::copy(std::begin(weight_values), std::end(weight_values), weights.mutable_data());

    float fused_output[m * n] = {0};
    float reference[m * n] = {0};

    OneDNNGemmKernel kernel(&weights);

    ASSERT_TRUE(kernel.multiply_with_softmax(
        A,
        fused_output,
        m,
        n,
        k,
        true,
        1,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply(
        A,
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

TEST(Test__OneDNNGemmKernel, WeightFusedMatmulColumnSoftmaxMatchesReference)
{
    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int k = 3;

    const float A[m * k] = {
        0.1f, -0.2f, 0.3f,
        -0.6f, 0.4f, 0.9f,
        0.8f, -0.7f, 0.2f};

    const float weight_values[n * k] = {
        -0.3f, 0.5f, -0.1f,
        0.7f, -0.4f, 0.2f};

    FP32Tensor weights({static_cast<size_t>(n), static_cast<size_t>(k)});
    std::copy(std::begin(weight_values), std::end(weight_values), weights.mutable_data());

    float fused_output[m * n] = {0};
    float reference[m * n] = {0};

    OneDNNGemmKernel kernel(&weights);

    ASSERT_TRUE(kernel.multiply_with_softmax(
        A,
        fused_output,
        m,
        n,
        k,
        true,
        0,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply(
        A,
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

TEST(Test__OneDNNGemmKernel, WeightFusedMatmulLargeShapeSoftmaxMatchesReference)
{
    constexpr int m = 6;
    constexpr int n = 7;
    constexpr int k = 12;

    std::vector<float> activations(static_cast<size_t>(m) * static_cast<size_t>(k));
    std::vector<float> weight_values(static_cast<size_t>(n) * static_cast<size_t>(k));

    for (size_t idx = 0; idx < activations.size(); ++idx)
    {
        activations[idx] = std::sin(static_cast<float>(idx) * 0.09f) * 0.6f + std::cos(static_cast<float>(idx) * 0.04f) * 0.2f;
    }
    for (size_t idx = 0; idx < weight_values.size(); ++idx)
    {
        weight_values[idx] = std::cos(static_cast<float>(idx) * 0.12f) * 0.5f - std::sin(static_cast<float>(idx) * 0.03f) * 0.1f;
    }

    FP32Tensor weights({static_cast<size_t>(n), static_cast<size_t>(k)});
    std::copy(weight_values.begin(), weight_values.end(), weights.mutable_data());

    std::vector<float> fused_output(static_cast<size_t>(m) * static_cast<size_t>(n));
    std::vector<float> reference(static_cast<size_t>(m) * static_cast<size_t>(n));

    OneDNNGemmKernel kernel(&weights);

    ASSERT_TRUE(kernel.multiply_with_softmax(
        activations.data(),
        fused_output.data(),
        m,
        n,
        k,
        true,
        1,
        nullptr,
        -1));

    ASSERT_TRUE(kernel.multiply(
        activations.data(),
        reference.data(),
        m,
        n,
        k,
        true,
        1.0f,
        0.0f,
        nullptr,
        -1));

    apply_rowwise_softmax(reference.data(), m, n);

    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(fused_output[static_cast<size_t>(i)], reference[static_cast<size_t>(i)], 5e-5f) << "Mismatch at idx " << i;
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
    EXPECT_FALSE(kernel.multiply_activations_with_softmax(
        A,
        B,
        C,
        m,
        n,
        k,
        true,
        2,
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
        C,
        m,
        n,
        k,
        true,
        2,
        nullptr,
        -1));
}

#else // HAVE_ONEDNN

TEST(Test__OneDNNGemmKernel, SkippedWithoutOneDNN)
{
    GTEST_SKIP() << "OneDNN backend not available";
}

#endif
