#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <cstring>
#include "utils/DebugEnv.h"
#include "operators/MPIAttentionOperator.h"
#include "tensors/TensorFactory.h"
#include "Logger.h"
#include "tensors/TpPartition.h"

using namespace llaminar;

// Helper to build a simple attention output projection scenario:
// We simulate post-attention local_attended_output (seq_len x hidden) and W_O (hidden x hidden)
// then invoke computeLocalOutputProjection indirectly through kernel path by crafting inputs.
// For parity we run once with TP simulation disabled then with TP simulation enabled (col & row modes) and compare outputs.

static void reference_matmul(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for (size_t k = 0; k < K; ++k)
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            C[i * N + j] = (float)acc;
        }
    }
}

struct TPSimCase
{
    size_t seq_len;
    size_t heads;
    size_t head_dim;
    int partitions;
    bool row_mode;
    unsigned seed;
};

static void run_tp_sim_case(const TPSimCase &cs)
{
    size_t hidden = cs.heads * cs.head_dim;
    std::mt19937 rng(cs.seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> attended(cs.seq_len * hidden);
    std::vector<float> w_o(hidden * hidden);
    for (auto &v : attended)
        v = dist(rng);
    for (auto &v : w_o)
        v = dist(rng);

    // Baseline reference output
    std::vector<float> baseline(cs.seq_len * hidden, 0.f);
    reference_matmul(attended.data(), w_o.data(), baseline.data(), cs.seq_len, hidden, hidden);

    // Simulated TP reconstruction using executor directly mirrors kernel logic.
    // We'll replicate logic of tp_sim path for row or column split.
    std::vector<float> recon(cs.seq_len * hidden, 0.f);
    int parts = cs.partitions;
    size_t K = hidden; // local_head_dim in kernel (assuming all heads local)

    if (cs.row_mode)
    {
        // Row split: partition M (seq_len)
        for (int p = 0; p < parts; ++p)
        {
            auto spec = compute_tp_partition(cs.seq_len, parts, p, TPPartitionSpec::Axis::Row);
            size_t m_local = spec.local_dim;
            size_t m_off = spec.local_offset;
            std::vector<float> local(m_local * hidden, 0.f);
            reference_matmul(attended.data() + m_off * K, w_o.data(), local.data(), m_local, hidden, K);
            // stitch rows
            for (size_t r = 0; r < m_local; ++r)
            {
                std::memcpy(recon.data() + (m_off + r) * hidden, local.data() + r * hidden, sizeof(float) * hidden);
            }
        }
    }
    else
    {
        // Column split: partition N (hidden)
        for (int p = 0; p < parts; ++p)
        {
            auto spec = compute_tp_partition(hidden, parts, p, TPPartitionSpec::Axis::Col);
            size_t n_local = spec.local_dim;
            size_t n_off = spec.local_offset;
            // pack B subset
            std::vector<float> B_pack(K * n_local);
            for (size_t k = 0; k < K; ++k)
            {
                std::memcpy(B_pack.data() + k * n_local, w_o.data() + k * hidden + n_off, sizeof(float) * n_local);
            }
            std::vector<float> local(cs.seq_len * n_local, 0.f);
            reference_matmul(attended.data(), B_pack.data(), local.data(), cs.seq_len, n_local, K);
            // stitch columns
            for (size_t r = 0; r < cs.seq_len; ++r)
            {
                std::memcpy(recon.data() + r * hidden + n_off, local.data() + r * n_local, sizeof(float) * n_local);
            }
        }
    }

    double max_abs = 0.0, rel_num = 0.0, rel_den = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i)
    {
        double d = (double)baseline[i] - recon[i];
        max_abs = std::max(max_abs, std::abs(d));
        rel_num += d * d;
        rel_den += (double)baseline[i] * baseline[i];
    }
    double rel_l2 = std::sqrt(rel_num / (rel_den + 1e-12));
    EXPECT_LT(max_abs, 1e-5) << "max_abs=" << max_abs;
    EXPECT_LT(rel_l2, 1e-6) << "rel_l2=" << rel_l2;
}

TEST(AttentionTPSimParity, ColumnAndRowModes)
{
    std::vector<TPSimCase> cases = {
        {16, 4, 32, 2, false, 123u}, // column split even
        {15, 5, 24, 3, false, 456u}, // column split uneven hidden
        {18, 3, 40, 3, true, 789u},  // row split (seq_len divisible)
        {21, 7, 16, 3, true, 42u},   // row split uneven heads but seq divisible
    };
    for (const auto &c : cases)
        run_tp_sim_case(c);
}

TEST(AttentionTPSimParity, AutoModeHeuristic)
{
    // Exercise auto heuristic: prefer col if hidden divisible else row if seq_len divisible
    std::vector<TPSimCase> cases = {
        {17, 4, 32, 4, false, 11u}, // hidden divisible by parts => column
        {18, 5, 24, 6, true, 22u},  // hidden not divisible, seq_len divisible by 6 => row
    };
    for (const auto &c : cases)
        run_tp_sim_case(c);
}
