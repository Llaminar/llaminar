#include <gtest/gtest.h>
#include "tensors/TpPartition.h"
#include <vector>
#include <numeric>
#include <cmath>
#include <cstring>

using namespace llaminar;

static bool ref_matmul(const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K)
{
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = (float)acc;
        }
    }
    return true;
}

TEST(TPSplitters, ColumnSplitParity)
{
    const size_t M = 5, K = 7, N = 11;
    int tp_size = 3; // uneven split
    std::vector<float> A(M * K), B(K * N);
    std::iota(A.begin(), A.end(), 1.f);
    std::iota(B.begin(), B.end(), 0.f);
    std::vector<float> full(M * N, 0.f);
    ref_matmul(A.data(), B.data(), full.data(), M, N, K);

    std::vector<float> recon(M * N, 0.f);
    size_t col_offset = 0;
    for (int r = 0; r < tp_size; ++r)
    {
        ColumnSplitMatmulSplitter splitter(N, tp_size, r, ref_matmul);
        auto spec = splitter.specB();
        std::vector<float> local(M * spec.local_dim, 0.f);
        MatmulSplitter::Args args{A.data(), B.data(), local.data(), M, spec.local_dim, K};
        ASSERT_TRUE(splitter.run(args));
        // stitch
        for (size_t i = 0; i < M; ++i)
        {
            std::memcpy(recon.data() + i * N + spec.local_offset, local.data() + i * spec.local_dim, sizeof(float) * spec.local_dim);
        }
        col_offset += spec.local_dim;
    }
    double max_abs = 0, rel_l2_num = 0, rel_l2_den = 0;
    for (size_t i = 0; i < full.size(); ++i)
    {
        double d = full[i] - recon[i];
        max_abs = std::max(max_abs, std::abs(d));
        rel_l2_num += d * d;
        rel_l2_den += (double)full[i] * full[i];
    }
    double rel_l2 = std::sqrt(rel_l2_num / rel_l2_den);
    EXPECT_LT(max_abs, 1e-6) << "max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-7) << "rel_l2=" << rel_l2;
}

TEST(TPSplitters, RowSplitParity)
{
    const size_t M = 13, K = 5, N = 4;
    int tp_size = 4;
    std::vector<float> A(M * K), B(K * N);
    std::iota(A.begin(), A.end(), 1.f);
    std::iota(B.begin(), B.end(), 0.f);
    std::vector<float> full(M * N, 0.f);
    ref_matmul(A.data(), B.data(), full.data(), M, N, K);

    std::vector<float> recon(M * N, 0.f);
    size_t row_offset = 0;
    for (int r = 0; r < tp_size; ++r)
    {
        RowSplitMatmulSplitter splitter(M, tp_size, r, ref_matmul);
        auto spec = splitter.specA();
        std::vector<float> local(spec.local_dim * N, 0.f);
        // provide pointer into A at local offset
        const float *A_local = A.data() + spec.local_offset * K;
        MatmulSplitter::Args args{A_local, B.data(), local.data(), spec.local_dim, N, K};
        ASSERT_TRUE(splitter.run(args));
        for (size_t i = 0; i < spec.local_dim; ++i)
        {
            std::memcpy(recon.data() + (spec.local_offset + i) * N, local.data() + i * N, sizeof(float) * N);
        }
        row_offset += spec.local_dim;
    }
    double max_abs = 0, rel_l2_num = 0, rel_l2_den = 0;
    for (size_t i = 0; i < full.size(); ++i)
    {
        double d = full[i] - recon[i];
        max_abs = std::max(max_abs, std::abs(d));
        rel_l2_num += d * d;
        rel_l2_den += (double)full[i] * full[i];
    }
    double rel_l2 = std::sqrt(rel_l2_num / rel_l2_den);
    EXPECT_LT(max_abs, 1e-6) << "max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-7) << "rel_l2=" << rel_l2;
}
