// Basic tests for TPPartitionSpec and trivial splitter
#include <gtest/gtest.h>
#include "tensors/TpPartition.h"
#include <vector>

using namespace llaminar;

TEST(TPPartitionSpecTest, SinglePartitionIdentity)
{
    auto spec = compute_tp_partition(1024, 1, 0, TPPartitionSpec::Axis::Row);
    EXPECT_FALSE(spec.active());
    EXPECT_EQ(spec.local_dim, 1024u);
    EXPECT_EQ(spec.local_offset, 0u);
}

TEST(TPPartitionSpecTest, EvenSplit)
{
    auto s0 = compute_tp_partition(100, 4, 0, TPPartitionSpec::Axis::Col);
    auto s1 = compute_tp_partition(100, 4, 1, TPPartitionSpec::Axis::Col);
    auto s2 = compute_tp_partition(100, 4, 2, TPPartitionSpec::Axis::Col);
    auto s3 = compute_tp_partition(100, 4, 3, TPPartitionSpec::Axis::Col);
    EXPECT_EQ(s0.local_dim + s1.local_dim + s2.local_dim + s3.local_dim, 100u);
    EXPECT_EQ(s0.local_offset, 0u);
    EXPECT_EQ(s1.local_offset, s0.local_dim);
    EXPECT_EQ(s2.local_offset, s0.local_dim + s1.local_dim);
    EXPECT_EQ(s3.local_offset, s0.local_dim + s1.local_dim + s2.local_dim);
}

TEST(TPPartitionSpecTest, RemainderSplit)
{
    // 10 items across 3 ranks -> sizes 4,3,3
    auto s0 = compute_tp_partition(10, 3, 0, TPPartitionSpec::Axis::Row);
    auto s1 = compute_tp_partition(10, 3, 1, TPPartitionSpec::Axis::Row);
    auto s2 = compute_tp_partition(10, 3, 2, TPPartitionSpec::Axis::Row);
    EXPECT_EQ(s0.local_dim, 4u);
    EXPECT_EQ(s1.local_dim, 3u);
    EXPECT_EQ(s2.local_dim, 3u);
    EXPECT_EQ(s0.local_offset, 0u);
    EXPECT_EQ(s1.local_offset, 4u);
    EXPECT_EQ(s2.local_offset, 7u);
}

TEST(TPPartitionSpecTest, TrivialMatmulExecutes)
{
    // Simple reference matmul (row-major) C[M,N] = A[M,K] * B[K,N]
    auto ref = [](const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K)
    {
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                float acc = 0.f;
                for (size_t k = 0; k < K; ++k)
                    acc += A[i * K + k] * B[k * N + j];
                C[i * N + j] = acc;
            }
        }
        return true;
    };
    TrivialMatmulSplitter splitter(ref);
    const size_t M = 4, N = 3, K = 5;
    std::vector<float> A(M * K), B(K * N), C(M * N), Cref(M * N);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<float>(i % 7 - 3);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = static_cast<float>((i * 3) % 5 - 1);
    // Run through splitter
    MatmulSplitter::Args args{A.data(), B.data(), C.data(), M, N, K};
    ASSERT_TRUE(splitter.run(args));
    // Direct reference
    ASSERT_TRUE(ref(A.data(), B.data(), Cref.data(), M, N, K));
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_FLOAT_EQ(C[i], Cref[i]);
    }
}

// Simulate a row-partitioned matmul: split M across faux partitions, compute slices, stitch.
TEST(TPPartitionSpecTest, SimulatedRowPartitionParity)
{
    const size_t M = 17, N = 11, K = 9; // choose sizes not divisible by tp_size
    const int tp_size = 4;
    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_recon(M * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (float)((i * 7) % 13 - 6);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (float)((i * 5) % 11 - 5);
    auto ref = [&](float *C)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
            {
                float acc = 0.f;
                for (size_t k = 0; k < K; ++k)
                    acc += A[i * K + k] * B[k * N + j];
                C[i * N + j] = acc;
            }
    };
    ref(C_ref.data());
    for (int r = 0; r < tp_size; ++r)
    {
        auto spec = compute_tp_partition(M, tp_size, r, TPPartitionSpec::Axis::Row);
        for (size_t li = 0; li < spec.local_dim; ++li)
        {
            size_t gi = spec.local_offset + li;
            for (size_t j = 0; j < N; ++j)
            {
                float acc = 0.f;
                for (size_t k = 0; k < K; ++k)
                    acc += A[gi * K + k] * B[k * N + j];
                C_recon[gi * N + j] = acc;
            }
        }
    }
    for (size_t i = 0; i < C_ref.size(); ++i)
        EXPECT_FLOAT_EQ(C_ref[i], C_recon[i]);
}

// Simulate a column-partitioned matmul: split N, compute partial columns, stitch.
TEST(TPPartitionSpecTest, SimulatedColPartitionParity)
{
    const size_t M = 13, N = 19, K = 7;
    const int tp_size = 5;
    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_recon(M * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (float)((i * 3) % 17 - 8);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (float)((i * 11) % 19 - 9);
    auto ref = [&](float *C)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
            {
                float acc = 0.f;
                for (size_t k = 0; k < K; ++k)
                    acc += A[i * K + k] * B[k * N + j];
                C[i * N + j] = acc;
            }
    };
    ref(C_ref.data());
    for (int r = 0; r < tp_size; ++r)
    {
        auto spec = compute_tp_partition(N, tp_size, r, TPPartitionSpec::Axis::Col);
        for (size_t jlocal = 0; jlocal < spec.local_dim; ++jlocal)
        {
            size_t gj = spec.local_offset + jlocal;
            for (size_t i = 0; i < M; ++i)
            {
                float acc = 0.f;
                for (size_t k = 0; k < K; ++k)
                    acc += A[i * K + k] * B[k * N + gj];
                C_recon[i * N + gj] = acc;
            }
        }
    }
    for (size_t i = 0; i < C_ref.size(); ++i)
        EXPECT_FLOAT_EQ(C_ref[i], C_recon[i]);
}
