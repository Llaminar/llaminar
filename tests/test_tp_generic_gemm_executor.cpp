#include "../src/tensors/tp_generic_matmul_executor.h"
#include <gtest/gtest.h>
#include <random>
#include <numeric>
#include <cmath>

using namespace llaminar;

static bool ref_matmul(const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K)
{
    // Simple triple loop reference
    std::fill(C, C + M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m)
    {
        for (std::size_t k = 0; k < K; ++k)
        {
            float a = A[m * K + k];
            const float *brow = B + k * N;
            float *crow = C + m * N;
            for (std::size_t n = 0; n < N; ++n)
                crow[n] += a * brow[n];
        }
    }
    return true;
}

struct Case
{
    std::size_t M, N, K;
    int tp_size;
    TPGemmExecConfig::Mode mode;
};

class TPGemmExecutorParity : public ::testing::TestWithParam<Case>
{
};

TEST_P(TPGemmExecutorParity, ParityVsReference)
{
    auto c = GetParam();
    // Random deterministic input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> A(c.M * c.K), B(c.K * c.N), C_ref(c.M * c.N);
    for (auto &v : A)
        v = dist(rng);
    for (auto &v : B)
        v = dist(rng);

    ASSERT_TRUE(ref_matmul(A.data(), B.data(), C_ref.data(), c.M, c.N, c.K));

    // Run partitions
    std::vector<TPGemmLocalResult> parts;
    for (int r = 0; r < c.tp_size; ++r)
    {
        TPGemmExecConfig cfg;
        cfg.mode = c.mode;
        cfg.tp_size = c.tp_size;
        cfg.tp_rank = r;
        TPGemmExecutor exec(ref_matmul, cfg, c.M, c.N, c.K);
        auto res = exec.run(A.data(), B.data());
        parts.push_back(std::move(res));
    }

    std::vector<float> C_recon(c.M * c.N, 0.0f);
    if (c.mode == TPGemmExecConfig::Mode::Column)
    {
        TPGemmExecutor::reconstruct_columns(parts, C_recon.data(), c.M, c.N);
    }
    else
    {
        TPGemmExecutor::reconstruct_rows(parts, C_recon.data(), c.M, c.N);
    }

    // Compare
    double diff_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
    for (std::size_t i = 0; i < c.M * c.N; ++i)
    {
        double d = (double)C_recon[i] - (double)C_ref[i];
        diff_sq += d * d;
        ref_sq += (double)C_ref[i] * (double)C_ref[i];
        double absd = std::fabs(d);
        if (absd > max_abs)
            max_abs = absd;
    }
    double rel_l2 = (ref_sq > 0.0) ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
    EXPECT_LT(max_abs, 1e-5) << "max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-6) << "rel_l2=" << rel_l2;
}

INSTANTIATE_TEST_SUITE_P(
    Basic,
    TPGemmExecutorParity,
    ::testing::Values(
        Case{8, 16, 32, 1, TPGemmExecConfig::Mode::Column},  // no split
        Case{8, 31, 17, 3, TPGemmExecConfig::Mode::Column},  // uneven N
        Case{19, 16, 13, 4, TPGemmExecConfig::Mode::Row},    // uneven M
        Case{37, 53, 11, 5, TPGemmExecConfig::Mode::Column}, // prime dims column split
        Case{41, 29, 7, 6, TPGemmExecConfig::Mode::Row}      // prime dims row split
        ));
