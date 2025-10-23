#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <cstring>
#include "tensors/TpPartition.h"

using namespace llaminar;

static bool ref_matmul(const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K)
{
    // A: [M,K] row-major, B: [K,N] row-major, C: [M,N]
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

static void run_column_tp_parity(size_t seq_len, size_t hidden, int tp_size, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // Simulated attention context (already concatenated heads): [seq_len, hidden]
    std::vector<float> context(seq_len * hidden);
    for (auto &v : context)
        v = dist(rng);
    // Output projection W_O: [hidden, hidden]
    std::vector<float> W(hidden * hidden);
    for (auto &v : W)
        v = dist(rng);

    // Baseline full matmul
    std::vector<float> baseline(seq_len * hidden, 0.f);
    ref_matmul(context.data(), W.data(), baseline.data(), seq_len, hidden, hidden);

    // TP column split reconstruction
    std::vector<float> recon(seq_len * hidden, 0.f);
    for (int r = 0; r < tp_size; ++r)
    {
        ColumnSplitMatmulSplitter splitter(hidden, tp_size, r, ref_matmul);
        auto spec = splitter.specB();
        ASSERT_EQ(spec.global_dim, hidden);
        std::vector<float> local(seq_len * spec.local_dim, 0.f);
        MatmulSplitter::Args args{context.data(), W.data(), local.data(), seq_len, spec.local_dim, hidden};
        ASSERT_TRUE(splitter.run(args));
        // Stitch columns back
        for (size_t i = 0; i < seq_len; ++i)
        {
            std::memcpy(recon.data() + i * hidden + spec.local_offset, local.data() + i * spec.local_dim, sizeof(float) * spec.local_dim);
        }
    }

    double max_abs = 0.0, rel_l2_num = 0.0, rel_l2_den = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i)
    {
        double diff = (double)baseline[i] - (double)recon[i];
        max_abs = std::max(max_abs, std::abs(diff));
        rel_l2_num += diff * diff;
        rel_l2_den += (double)baseline[i] * baseline[i];
    }
    double rel_l2 = std::sqrt(rel_l2_num / rel_l2_den);
    EXPECT_LT(max_abs, 1e-5) << "max_abs=" << max_abs << " seq=" << seq_len << " hidden=" << hidden << " tp_size=" << tp_size;
    EXPECT_LT(rel_l2, 1e-6) << "rel_l2=" << rel_l2;
}

TEST(AttentionTPParity, UnevenHiddenMultiSeeds)
{
    std::vector<unsigned> seeds{0u, 1u, 42u, 1234u, 99991u};
    for (auto s : seeds)
    {
        run_column_tp_parity(7, 65, 3, s);
    }
}

TEST(AttentionTPParity, EvenHiddenMultiSeeds)
{
    std::vector<unsigned> seeds{5u, 6u, 77u, 5678u, 2025u};
    for (auto s : seeds)
    {
        run_column_tp_parity(8, 64, 4, s);
    }
}

TEST(AttentionTPParity, MixedDimsRandomized)
{
    // Additional quick randomized coverage for varying seq_len & tp_size combinations
    struct Case
    {
        size_t S;
        size_t H;
        int tp;
        unsigned seed;
    };
    std::vector<Case> cases = {
        {5, 33, 2, 11u},  // small uneven hidden
        {6, 96, 3, 12u},  // divisible by 3
        {9, 70, 4, 13u},  // uneven vs tp=4
        {4, 128, 5, 14u}, // prime tp size style partition (simulated)
    };
    for (const auto &c : cases)
    {
        run_column_tp_parity(c.S, c.H, c.tp, c.seed);
    }
}
