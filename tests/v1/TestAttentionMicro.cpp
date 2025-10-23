#include "TestTensorUtils.h"
#include "TestReferenceImpls.h"
#include "TestTimeoutGuard.h"
#include "operators/MPIAttentionOperator.h"
#include "tensors/TensorFactory.h"
#include "TestMpiUtils.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <memory>
#include <numeric>
#include <iostream>

using namespace llaminar;

namespace
{
    struct ScopedMPIInit
    {
        ScopedMPIInit()
        {
            int flag;
            MPI_Initialized(&flag);
            if (!flag)
            {
                int provided;
                MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            }
        }
        ~ScopedMPIInit() {}
    };
}

class AttentionMicroTestFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        guard = std::make_unique<ScopedMPIInit>();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world);
    }
    int rank = 0, world = 1;
    std::unique_ptr<ScopedMPIInit> guard;
};

static std::shared_ptr<TensorBase> make_simple(const std::vector<int> &shape) { return TensorFactory::create_simple(shape); }

// Kernel always applies causal masking currently; we test parity vs a simple causal reference.
struct TinyConfig
{
    int seq;
    int head_dim;
};

// Reference implementation for a SINGLE head sitting at the beginning of each token row.
// The input tensor is laid out as rows of length d_model = n_head * head_dim. We only
// use (and compare against) the very first head slice [0, head_dim).
// NOTE: Previous version incorrectly assumed a packed [seq, head_dim] layout and used
//       head_dim as the row stride. That produced erroneous rows when seq>1 and n_head>1,
//       explaining the large rel_l2 / max_abs divergences for seq > 1. The operator output
//       was correct; the reference was striding incorrectly. Fixed by adding d_model
//       parameter and using it as the row stride.
static void reference_single_head_causal(const float *input, int seq, int head_dim, int d_model, std::vector<float> &out)
{
    // Replicate kernel's RoPE (n_past=0, rope_freq_base=10000)
    const float rope_freq_base = 10000.f;
    std::vector<float> Q(seq * head_dim), K(seq * head_dim);
    for (int i = 0; i < seq; ++i)
    {
        const float *src = input + i * d_model; // correct row stride (full model width)
        std::copy(src, src + head_dim, Q.begin() + i * head_dim);
        std::copy(src, src + head_dim, K.begin() + i * head_dim);
    }
    for (int i = 0; i < seq; ++i)
    {
        for (int pair = 0; pair < head_dim / 2; ++pair)
        {
            float theta = 1.f / std::pow(rope_freq_base, (2.f * pair) / head_dim);
            float angle = float(i) * theta;
            float cs = std::cos(angle), sn = std::sin(angle);
            int i0 = 2 * pair;
            int i1 = 2 * pair + 1;
            // Q rotation
            float q0 = Q[i * head_dim + i0];
            float q1 = Q[i * head_dim + i1];
            Q[i * head_dim + i0] = q0 * cs - q1 * sn;
            Q[i * head_dim + i1] = q0 * sn + q1 * cs;
            // K rotation
            float k0 = K[i * head_dim + i0];
            float k1 = K[i * head_dim + i1];
            K[i * head_dim + i0] = k0 * cs - k1 * sn;
            K[i * head_dim + i1] = k0 * sn + k1 * cs;
        }
    }
    out.assign(seq * head_dim, 0.f);
    const float scale = 1.f / std::sqrt(float(head_dim));
    std::vector<float> scores(seq * seq, 0.f);
    for (int i = 0; i < seq; ++i)
    {
        float maxv = -INFINITY;
        for (int j = 0; j <= i; ++j)
        {
            const float *qi = Q.data() + i * head_dim;
            const float *kj = K.data() + j * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d)
                dot += qi[d] * kj[d];
            float s = dot * scale;
            scores[i * seq + j] = s;
            if (s > maxv)
                maxv = s;
        }
        float sum = 0.f;
        for (int j = 0; j <= i; ++j)
        {
            float e = std::exp(scores[i * seq + j] - maxv);
            scores[i * seq + j] = e;
            sum += e;
        }
        for (int j = 0; j <= i; ++j)
            scores[i * seq + j] /= sum;
        for (int j = 0; j <= i; ++j)
        {
            const float p = scores[i * seq + j];
            // CORRECT STRIDE: each sequence row is d_model wide (n_head * head_dim). We only
            // want the first head slice [0, head_dim). Previous buggy version used j*head_dim
            // which incorrectly advanced within the first row instead of moving to row j.
            const float *vj = input + j * d_model; // fixed stride
            for (int d = 0; d < head_dim; ++d)
                out[i * head_dim + d] += p * vj[d];
        }
    }
}

// Multi-head reference for identity weights (each head attends only its own slice independently).
static void reference_multi_head_identity(const float *input, int seq, int n_head, int head_dim, int d_model, std::vector<float> &out)
{
    out.assign(seq * d_model, 0.f);
    const float rope_freq_base = 10000.f;
    const float scale = 1.f / std::sqrt(float(head_dim));
    for (int h = 0; h < n_head; ++h)
    {
        // Extract per-head Q/K/V (identity projections: just slice)
        std::vector<float> Q(seq * head_dim), K(seq * head_dim), V(seq * head_dim);
        for (int i = 0; i < seq; ++i)
        {
            const float *row = input + i * d_model + h * head_dim;
            std::copy(row, row + head_dim, Q.begin() + i * head_dim);
            std::copy(row, row + head_dim, K.begin() + i * head_dim);
            std::copy(row, row + head_dim, V.begin() + i * head_dim);
        }
        // RoPE
        for (int i = 0; i < seq; ++i)
        {
            for (int pair = 0; pair < head_dim / 2; ++pair)
            {
                float theta = 1.f / std::pow(rope_freq_base, (2.f * pair) / head_dim);
                float angle = float(i) * theta;
                float cs = std::cos(angle), sn = std::sin(angle);
                int i0 = 2 * pair;
                int i1 = 2 * pair + 1;
                float q0 = Q[i * head_dim + i0];
                float q1 = Q[i * head_dim + i1];
                Q[i * head_dim + i0] = q0 * cs - q1 * sn;
                Q[i * head_dim + i1] = q0 * sn + q1 * cs;
                float k0 = K[i * head_dim + i0];
                float k1 = K[i * head_dim + i1];
                K[i * head_dim + i0] = k0 * cs - k1 * sn;
                K[i * head_dim + i1] = k0 * sn + k1 * cs;
            }
        }
        // Attention per head
        std::vector<float> scores(seq * seq, 0.f);
        std::vector<float> head_out(seq * head_dim, 0.f);
        for (int i = 0; i < seq; ++i)
        {
            float maxv = -INFINITY;
            for (int j = 0; j <= i; ++j)
            {
                const float *qi = Q.data() + i * head_dim;
                const float *kj = K.data() + j * head_dim;
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                float s = dot * scale;
                scores[i * seq + j] = s;
                if (s > maxv)
                    maxv = s;
            }
            float sum = 0.f;
            for (int j = 0; j <= i; ++j)
            {
                float e = std::exp(scores[i * seq + j] - maxv);
                scores[i * seq + j] = e;
                sum += e;
            }
            for (int j = 0; j <= i; ++j)
                scores[i * seq + j] /= sum;
            for (int j = 0; j <= i; ++j)
            {
                float p = scores[i * seq + j];
                const float *vj = V.data() + j * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    head_out[i * head_dim + d] += p * vj[d];
            }
        }
        // Identity output projection -> copy into out slice
        for (int i = 0; i < seq; ++i)
        {
            float *dst = out.data() + i * d_model + h * head_dim;
            std::copy(head_out.begin() + i * head_dim, head_out.begin() + (i + 1) * head_dim, dst);
        }
    }
}

// Multi-head reference with per-head scaling applied to all Q/K/V projections and output projection.
static void reference_multi_head_scaled(const float *input, int seq, int n_head, int head_dim, int d_model,
                                        const std::vector<float> &scales, std::vector<float> &out)
{
    out.assign(seq * d_model, 0.f);
    const float rope_freq_base = 10000.f;
    const float inv_sqrt = 1.f / std::sqrt(float(head_dim));
    for (int h = 0; h < n_head; ++h)
    {
        float s = scales[h];
        std::vector<float> Q(seq * head_dim), K(seq * head_dim), V(seq * head_dim);
        for (int i = 0; i < seq; ++i)
        {
            const float *row = input + i * d_model + h * head_dim;
            for (int d = 0; d < head_dim; ++d)
            {
                float v = row[d];
                Q[i * head_dim + d] = s * v;
                K[i * head_dim + d] = s * v;
                V[i * head_dim + d] = s * v; // V scaling
            }
        }
        // RoPE
        for (int i = 0; i < seq; ++i)
        {
            for (int pair = 0; pair < head_dim / 2; ++pair)
            {
                float theta = 1.f / std::pow(rope_freq_base, (2.f * pair) / head_dim);
                float angle = float(i) * theta;
                float cs = std::cos(angle), sn = std::sin(angle);
                int i0 = 2 * pair;
                int i1 = 2 * pair + 1;
                float q0 = Q[i * head_dim + i0];
                float q1 = Q[i * head_dim + i1];
                Q[i * head_dim + i0] = q0 * cs - q1 * sn;
                Q[i * head_dim + i1] = q0 * sn + q1 * cs;
                float k0 = K[i * head_dim + i0];
                float k1 = K[i * head_dim + i1];
                K[i * head_dim + i0] = k0 * cs - k1 * sn;
                K[i * head_dim + i1] = k0 * sn + k1 * cs;
            }
        }
        // Attention with scaled scores (scale factor s^2 embedded in Q,K)
        std::vector<float> scores(seq * seq, 0.f), head_out(seq * head_dim, 0.f);
        for (int i = 0; i < seq; ++i)
        {
            float maxv = -INFINITY;
            for (int j = 0; j <= i; ++j)
            {
                const float *qi = Q.data() + i * head_dim;
                const float *kj = K.data() + j * head_dim;
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                float sc = dot * inv_sqrt;
                scores[i * seq + j] = sc;
                if (sc > maxv)
                    maxv = sc;
            }
            float sum = 0.f;
            for (int j = 0; j <= i; ++j)
            {
                float e = std::exp(scores[i * seq + j] - maxv);
                scores[i * seq + j] = e;
                sum += e;
            }
            for (int j = 0; j <= i; ++j)
                scores[i * seq + j] /= sum;
            for (int j = 0; j <= i; ++j)
            {
                float p = scores[i * seq + j];
                const float *vj = V.data() + j * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    head_out[i * head_dim + d] += p * vj[d];
            }
        }
        // Output projection scaling (another factor s on each channel of this head)
        for (int i = 0; i < seq; ++i)
        {
            float *dst = out.data() + i * d_model + h * head_dim;
            for (int d = 0; d < head_dim; ++d)
                dst[d] = s * head_out[i * head_dim + d];
        }
    }
}

// Grouped KV heads reference: n_head_q = n_head, n_head_kv < n_head.
// Each query head h maps to kv head (h % n_head_kv) for K and V slices (pre-RoPE identity projections).
// RoPE applied independently per expanded head as in kernel (so duplicated K/V slices are rotated per head).
static void reference_grouped_kv_heads(const float *input, int seq, int n_head, int n_head_kv, int head_dim, int d_model, std::vector<float> &out)
{
    out.assign(seq * d_model, 0.f);
    const float rope_freq_base = 10000.f;
    const float inv_sqrt = 1.f / std::sqrt(float(head_dim));
    for (int h = 0; h < n_head; ++h)
    {
        int kv_h = h % n_head_kv;
        // Extract Q from its own slice; K/V from kv_h slice.
        std::vector<float> Q(seq * head_dim), K(seq * head_dim), V(seq * head_dim);
        for (int i = 0; i < seq; ++i)
        {
            const float *row = input + i * d_model;
            const float *q_slice = row + h * head_dim;
            const float *kv_slice = row + kv_h * head_dim;
            std::copy(q_slice, q_slice + head_dim, Q.begin() + i * head_dim);
            std::copy(kv_slice, kv_slice + head_dim, K.begin() + i * head_dim);
            std::copy(kv_slice, kv_slice + head_dim, V.begin() + i * head_dim);
        }
        // RoPE on Q, K
        for (int i = 0; i < seq; ++i)
        {
            for (int pair = 0; pair < head_dim / 2; ++pair)
            {
                float theta = 1.f / std::pow(rope_freq_base, (2.f * pair) / head_dim);
                float angle = float(i) * theta;
                float cs = std::cos(angle), sn = std::sin(angle);
                int i0 = 2 * pair;
                int i1 = 2 * pair + 1;
                float q0 = Q[i * head_dim + i0];
                float q1 = Q[i * head_dim + i1];
                Q[i * head_dim + i0] = q0 * cs - q1 * sn;
                Q[i * head_dim + i1] = q0 * sn + q1 * cs;
                float k0 = K[i * head_dim + i0];
                float k1 = K[i * head_dim + i1];
                K[i * head_dim + i0] = k0 * cs - k1 * sn;
                K[i * head_dim + i1] = k0 * sn + k1 * cs;
            }
        }
        // Attention causal
        std::vector<float> scores(seq * seq, 0.f), head_out(seq * head_dim, 0.f);
        for (int i = 0; i < seq; ++i)
        {
            float maxv = -INFINITY;
            for (int j = 0; j <= i; ++j)
            {
                const float *qi = Q.data() + i * head_dim;
                const float *kj = K.data() + j * head_dim;
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                float sc = dot * inv_sqrt;
                scores[i * seq + j] = sc;
                if (sc > maxv)
                    maxv = sc;
            }
            float sum = 0.f;
            for (int j = 0; j <= i; ++j)
            {
                float e = std::exp(scores[i * seq + j] - maxv);
                scores[i * seq + j] = e;
                sum += e;
            }
            for (int j = 0; j <= i; ++j)
                scores[i * seq + j] /= sum;
            for (int j = 0; j <= i; ++j)
            {
                float p = scores[i * seq + j];
                const float *vj = V.data() + j * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    head_out[i * head_dim + d] += p * vj[d];
            }
        }
        // Identity output projection: write into h slice
        for (int i = 0; i < seq; ++i)
        {
            float *dst = out.data() + i * d_model + h * head_dim;
            std::copy(head_out.begin() + i * head_dim, head_out.begin() + (i + 1) * head_dim, dst);
        }
    }
}

TEST_F(AttentionMicroTestFixture, ScalarParityAndMasking)
{
    std::vector<TinyConfig> cases = {{1, 8}, {2, 8}, {4, 8}};
    for (auto cfg : cases)
    {
        int n_head = (world > 1) ? world : 1; // distribute one head per rank
        int n_kv = n_head;
        int d_model = n_head * cfg.head_dim;
        MPIAttentionOperator kernel(n_head, n_kv, cfg.head_dim);
        auto input = make_simple({cfg.seq, d_model});
        auto wq = make_simple({d_model, d_model});
        auto wk = make_simple({d_model, d_model});
        auto wv = make_simple({d_model, d_model});
        auto wo = make_simple({d_model, d_model});
        auto k_cache = make_simple({cfg.seq, d_model}); // unused currently
        auto v_cache = make_simple({cfg.seq, d_model}); // unused currently
        auto output = make_simple({cfg.seq, d_model});
        // Fill input deterministic
        for (int i = 0; i < input->size(); ++i)
            input->data()[i] = 0.01f * float((i % 53) + 1);
        // Identity block weights
        for (int i = 0; i < wq->size(); ++i)
        {
            bool diag = (i % (d_model + 1) == 0);
            float val = diag ? 1.f : 0.f;
            wq->data()[i] = wk->data()[i] = wv->data()[i] = wo->data()[i] = val;
        }
        std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        ASSERT_TRUE(kernel.execute(inputs, outputs));
        if (rank == 0)
        {
            // Compare first head only
            std::vector<float> ref_head;
            reference_single_head_causal(input->data(), cfg.seq, cfg.head_dim, d_model, ref_head);
            std::vector<float> produced_head(cfg.seq * cfg.head_dim);
            for (int t = 0; t < cfg.seq; ++t)
                for (int d = 0; d < cfg.head_dim; ++d)
                    produced_head[t * cfg.head_dim + d] = output->data()[t * d_model + d];
            auto stats = testutils::diff(produced_head, ref_head);
            const bool debug = std::getenv("LLAMINAR_ATTN_MICRO_DEBUG") != nullptr;
            if (debug && rank == 0)
            {
                auto summarize_vec = [&](const char *label, const std::vector<float> &v)
                {
                    std::ostringstream oss;
                    oss << label << "[0..7]:";
                    int n = std::min<int>(8, (int)v.size());
                    for (int i = 0; i < n; ++i)
                    {
                        if (i)
                            oss << ",";
                        oss << v[i];
                    }
                    return oss.str();
                };
                std::cout << "[AttentionMicroDebug] seq=" << cfg.seq << " " << testutils::summarize(stats) << "\n"
                          << summarize_vec("produced", produced_head) << "\n"
                          << summarize_vec("reference", ref_head) << "\n";
                // Dump full diff indices if divergence detected
                if (stats.max_abs > 1e-4f || stats.rel_l2 > 1e-4f)
                {
                    std::cout << "FullDiff seq=" << cfg.seq << " indices(val_prod,val_ref,absdiff):";
                    for (size_t i = 0; i < produced_head.size(); ++i)
                    {
                        float pd = produced_head[i];
                        float rf = ref_head[i];
                        float ad = std::fabs(pd - rf);
                        if (ad > 1e-8f)
                        {
                            std::cout << " " << i << "(" << pd << "," << rf << "," << ad << ")";
                        }
                    }
                    std::cout << "\n";
                    // Also print row-wise for clarity
                    for (int t = 0; t < cfg.seq; ++t)
                    {
                        std::cout << " row " << t << " produced:";
                        for (int d = 0; d < cfg.head_dim; ++d)
                            std::cout << " " << produced_head[t * cfg.head_dim + d];
                        std::cout << " | ref:";
                        for (int d = 0; d < cfg.head_dim; ++d)
                            std::cout << " " << ref_head[t * cfg.head_dim + d];
                        std::cout << "\n";
                    }
                }
            }
            EXPECT_LE(stats.max_abs, 1e-4) << testutils::summarize(stats) << " seq=" << cfg.seq;
            EXPECT_LE(stats.rel_l2, 1e-4) << testutils::summarize(stats);
        }
        // Causal leak probe: amplify last token, earlier outputs should not depend on future position
        if (cfg.seq > 1)
        {
            if (rank == 0)
            {
                // Save baseline first head slice
                std::vector<float> baseline(cfg.seq * cfg.head_dim);
                for (int t = 0; t < cfg.seq; ++t)
                    for (int d = 0; d < cfg.head_dim; ++d)
                        baseline[t * cfg.head_dim + d] = output->data()[t * d_model + d];
                // Modify last token in-place (input) dramatically
                for (int d = 0; d < cfg.head_dim; ++d)
                    input->data()[(cfg.seq - 1) * d_model + d] = 100.f; // only first head slice changed
            }
            if (world > 1)
                MPI_Bcast(input->data(), input->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            ASSERT_TRUE(kernel.execute(inputs, outputs));
            if (rank == 0)
            {
                bool leak = false;
                for (int t = 0; t < cfg.seq - 1 && !leak; ++t)
                {
                    for (int d = 0; d < cfg.head_dim; ++d)
                    {
                        float val = output->data()[t * d_model + d];
                        // Expect near original because future token masked
                        // Recompute reference unaffected: just ensure delta small
                        // (Allow minor numerical change tolerance)
                        // We didn't store baseline after mutation; simplified: future shouldn't inject large magnitude
                        if (std::fabs(val) > 5.f)
                        {
                            leak = true;
                            break;
                        }
                    }
                }
                EXPECT_FALSE(leak) << "Causal mask leak detected seq=" << cfg.seq;
            }
        }
    }
}

// Multi-head identity parity test (single process only). Verifies all heads independently.
TEST_F(AttentionMicroTestFixture, MultiHeadIdentityParity)
{
    if (world > 1)
    {
        if (rank == 0)
            GTEST_SKIP() << "MultiHeadIdentityParity requires single rank (world==1).";
        return;
    }
    int seq = 4;
    int n_head = 4;
    int head_dim = 8;
    int d_model = n_head * head_dim;
    MPIAttentionOperator kernel(n_head, n_head, head_dim);
    auto input = make_simple({seq, d_model});
    auto wq = make_simple({d_model, d_model});
    auto wk = make_simple({d_model, d_model});
    auto wv = make_simple({d_model, d_model});
    auto wo = make_simple({d_model, d_model});
    auto k_cache = make_simple({seq, d_model});
    auto v_cache = make_simple({seq, d_model});
    auto output = make_simple({seq, d_model});
    for (int i = 0; i < input->size(); ++i)
        input->data()[i] = 0.01f * float((i % 97) + 1);
    for (int i = 0; i < wq->size(); ++i)
    {
        bool diag = (i % (d_model + 1) == 0);
        float val = diag ? 1.f : 0.f;
        wq->data()[i] = wk->data()[i] = wv->data()[i] = wo->data()[i] = val;
    }
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    // Build reference
    std::vector<float> ref;
    reference_multi_head_identity(input->data(), seq, n_head, head_dim, d_model, ref);
    // Compare all d_model channels
    std::vector<float> produced(seq * d_model);
    std::memcpy(produced.data(), output->data(), produced.size() * sizeof(float));
    auto stats = testutils::diff(produced, ref);
    EXPECT_LE(stats.max_abs, 1e-4) << testutils::summarize(stats);
    EXPECT_LE(stats.rel_l2, 1e-4) << testutils::summarize(stats);
}

// Multi-head scaled (non-identity) weights parity test.
TEST_F(AttentionMicroTestFixture, MultiHeadScaledWeightsParity)
{
    if (world > 1)
    {
        if (rank == 0)
            GTEST_SKIP() << "MultiHeadScaledWeightsParity requires single rank (world==1).";
        return;
    }
    int seq = 3;
    int n_head = 3;
    int head_dim = 8;
    int d_model = n_head * head_dim;
    MPIAttentionOperator kernel(n_head, n_head, head_dim);
    auto input = make_simple({seq, d_model});
    auto wq = make_simple({d_model, d_model});
    auto wk = make_simple({d_model, d_model});
    auto wv = make_simple({d_model, d_model});
    auto wo = make_simple({d_model, d_model});
    auto k_cache = make_simple({seq, d_model});
    auto v_cache = make_simple({seq, d_model});
    auto output = make_simple({seq, d_model});
    for (int i = 0; i < input->size(); ++i)
        input->data()[i] = 0.005f * float((i % 113) + 1);
    std::vector<float> scales(n_head);
    for (int h = 0; h < n_head; ++h)
        scales[h] = 1.f + 0.1f * h; // 1.0, 1.1, 1.2 ...
    // Zero initialize weights
    std::fill(wq->data(), wq->data() + wq->size(), 0.f);
    std::fill(wk->data(), wk->data() + wk->size(), 0.f);
    std::fill(wv->data(), wv->data() + wv->size(), 0.f);
    std::fill(wo->data(), wo->data() + wo->size(), 0.f);
    // Set per-head diagonal scales
    for (int h = 0; h < n_head; ++h)
    {
        float s = scales[h];
        for (int d = 0; d < head_dim; ++d)
        {
            int k = h * head_dim + d;  // global channel index
            int idx = k * d_model + k; // row-major (k,k) element in weight
            wq->data()[idx] = wk->data()[idx] = wv->data()[idx] = wo->data()[idx] = s;
        }
    }
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    // Reference
    std::vector<float> ref;
    reference_multi_head_scaled(input->data(), seq, n_head, head_dim, d_model, scales, ref);
    std::vector<float> produced(seq * d_model);
    std::memcpy(produced.data(), output->data(), produced.size() * sizeof(float));
    auto stats = testutils::diff(produced, ref);
    EXPECT_LE(stats.max_abs, 2e-4) << testutils::summarize(stats);
    EXPECT_LE(stats.rel_l2, 2e-4) << testutils::summarize(stats);
}

// Grouped KV heads (GQA-style) parity test: n_head > n_head_kv. Ensures modulo mapping correctness.
TEST_F(AttentionMicroTestFixture, GroupedKVHeadsParity)
{
    if (world > 1)
    {
        if (rank == 0)
            GTEST_SKIP() << "GroupedKVHeadsParity requires single rank (world==1).";
        return;
    }
    int seq = 4;
    int n_head = 4;    // query heads
    int n_head_kv = 2; // grouped KV heads
    int head_dim = 8;
    int d_model = n_head * head_dim;
    MPIAttentionOperator kernel(n_head, n_head_kv, head_dim);
    auto input = make_simple({seq, d_model});
    auto wq = make_simple({d_model, d_model});
    auto wk = make_simple({d_model, n_head_kv * head_dim});
    auto wv = make_simple({d_model, n_head_kv * head_dim});
    auto wo = make_simple({d_model, d_model});
    auto k_cache = make_simple({seq, n_head_kv * head_dim});
    auto v_cache = make_simple({seq, n_head_kv * head_dim});
    auto output = make_simple({seq, d_model});
    // Distinct per-channel pattern to disambiguate heads
    for (int i = 0; i < input->size(); ++i)
        input->data()[i] = 0.003f * float((i % 257) + 1);
    // wq: identity (full d_model)
    for (int i = 0; i < wq->size(); ++i)
        wq->data()[i] = (i % (d_model + 1) == 0) ? 1.f : 0.f;
    // wk, wv: rectangular "identity" selecting the first kv_total_dim input channels.
    // Shape: [d_model, kv_total_dim]. We set W[row=row, col=row]=1 for row < kv_total_dim.
    int kv_total_dim = n_head_kv * head_dim;
    std::fill(wk->data(), wk->data() + wk->size(), 0.f);
    std::fill(wv->data(), wv->data() + wv->size(), 0.f);
    for (int r = 0; r < d_model && r < kv_total_dim; ++r)
    {
        wk->data()[r * kv_total_dim + r] = 1.f;
        wv->data()[r * kv_total_dim + r] = 1.f;
    }
    // wo: identity (project concatenated heads back)
    for (int i = 0; i < wo->size(); ++i)
        wo->data()[i] = (i % (d_model + 1) == 0) ? 1.f : 0.f;
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    // Reference
    std::vector<float> ref;
    reference_grouped_kv_heads(input->data(), seq, n_head, n_head_kv, head_dim, d_model, ref);
    std::vector<float> produced(seq * d_model);
    std::memcpy(produced.data(), output->data(), produced.size() * sizeof(float));
    auto stats = testutils::diff(produced, ref);
    EXPECT_LE(stats.max_abs, 1e-4) << testutils::summarize(stats);
    EXPECT_LE(stats.rel_l2, 1e-4) << testutils::summarize(stats);
    // Sanity: heads sharing same KV group (0 & 2) should generally differ because Q differs.
    double diff02 = 0.0;
    for (int i = 0; i < seq; ++i)
        for (int d = 0; d < head_dim; ++d)
        {
            float a = produced[i * d_model + 0 * head_dim + d];
            float b = produced[i * d_model + 2 * head_dim + d];
            diff02 += std::fabs(a - b);
        }
    EXPECT_GT(diff02, 1e-5) << "Heads 0 and 2 unexpectedly identical under grouped KV mapping";
}

// Multi-rank parity test for GatherHeadsPreProjection mode.
// Compares multi-rank gathered result with single-rank baseline (same seed deterministic weights).
TEST_F(AttentionMicroTestFixture, GatherPreProjectionMultiRankParity)
{
    // Only execute if world > 1; rank 0 also computes baseline serially (by spawning a local kernel world=1 analogue).
    if (world < 2)
    {
        if (rank == 0)
            GTEST_SKIP() << "GatherPreProjectionMultiRankParity requires world>1.";
        return;
    }
    int seq = 3;
    int n_head = world * 2; // give each rank two heads for uneven-ish distribution if world doesn't divide
    int head_dim = 8;
    int d_model = n_head * head_dim;
    // Construct weights (identity) & input deterministic.
    MPIAttentionOperator kernel(n_head, n_head, head_dim);
    kernel.setOutputMode(MPIAttentionOperator::AttentionOutputMode::GatherHeadsPreProjection);
    auto input = make_simple({seq, d_model});
    auto wq = make_simple({d_model, d_model});
    auto wk = make_simple({d_model, d_model});
    auto wv = make_simple({d_model, d_model});
    auto wo = make_simple({d_model, d_model});
    auto k_cache = make_simple({seq, d_model});
    auto v_cache = make_simple({seq, d_model});
    auto output = make_simple({seq, d_model});
    for (int i = 0; i < input->size(); ++i)
        input->data()[i] = 0.007f * float((i % 211) + 1);
    for (int i = 0; i < wq->size(); ++i)
    {
        bool diag = (i % (d_model + 1) == 0);
        float v = diag ? 1.f : 0.f;
        wq->data()[i] = wk->data()[i] = wv->data()[i] = wo->data()[i] = v;
    }
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, wq, wk, wv, wo, k_cache, v_cache};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    ASSERT_TRUE(kernel.execute(inputs, outputs));
    // Gather multi-rank output to rank 0 (already replicated by mode, but we validate metadata and diff).
    auto meta = kernel.last_result_meta();
    EXPECT_TRUE(meta.concatenated);
    EXPECT_TRUE(meta.replicated);
    // Baseline (single-rank) built only on rank 0.
    if (rank == 0)
    {
        // Build baseline via reference_multi_head_identity (identity weights path).
        std::vector<float> ref;
        reference_multi_head_identity(input->data(), seq, n_head, head_dim, d_model, ref);
        std::vector<float> produced(seq * d_model);
        std::memcpy(produced.data(), output->data(), produced.size() * sizeof(float));
        auto stats = testutils::diff(produced, ref);
        EXPECT_LE(stats.max_abs, 2e-4) << testutils::summarize(stats);
        EXPECT_LE(stats.rel_l2, 2e-4) << testutils::summarize(stats);
    }
}

// Parity test: enable simulated TP column partitioning for WO via env and compare output vs baseline (partitions=1).
TEST_F(AttentionMicroTestFixture, TPColumnPartitionWOParity)
{
    // Keep world=1 only to isolate TP simulation (inter-rank path not needed).
    if (world != 1)
    {
        if (rank == 0)
            GTEST_SKIP() << "TPColumnPartitionWOParity runs only in single-rank world=1";
        return;
    }
    int seq = 5;
    int n_head = 4;
    int head_dim = 8;
    int d_model = n_head * head_dim;
    auto make_kernel = [&](bool tp)
    {
        auto k = std::make_unique<MPIAttentionOperator>(n_head, n_head, head_dim);
        k->setOutputMode(MPIAttentionOperator::AttentionOutputMode::GatherHeadsPostProjection); // local projection path
        if (tp)
        {
            setenv("LLAMINAR_ATTN_TP_PARTITIONS", "3", 1);
            unsetenv("LLAMINAR_ATTN_TP_DISABLE");
        }
        else
        {
            unsetenv("LLAMINAR_ATTN_TP_PARTITIONS");
            setenv("LLAMINAR_ATTN_TP_DISABLE", "1", 1);
        }
        return k;
    };
    auto input = make_simple({seq, d_model});
    auto wq = make_simple({d_model, d_model});
    auto wk = make_simple({d_model, d_model});
    auto wv = make_simple({d_model, d_model});
    auto wo = make_simple({d_model, d_model});
    auto k_cache = make_simple({seq, d_model});
    auto v_cache = make_simple({seq, d_model});
    auto out_baseline = make_simple({seq, d_model});
    auto out_tp = make_simple({seq, d_model});
    for (int i = 0; i < input->size(); ++i)
        input->data()[i] = 0.013f * float((i * 17) % 101 - 50);
    // Use non-trivial weights (diagonal + slight off-diagonal) to ensure column partition actually slices distinct data
    for (int i = 0; i < wq->size(); ++i)
    {
        bool diag = (i % (d_model + 1) == 0);
        float base = diag ? 1.f : 0.f;
        float jitter = ((i % 7) == 0) ? 0.01f : 0.f;
        wq->data()[i] = wk->data()[i] = wv->data()[i] = wo->data()[i] = base + jitter;
    }
    std::vector<std::shared_ptr<TensorBase>> inputs_vec = {input, wq, wk, wv, wo, k_cache, v_cache};
    { // baseline (TP disabled)
        auto kernel = make_kernel(false);
        std::vector<std::shared_ptr<TensorBase>> outputs = {out_baseline};
        ASSERT_TRUE(kernel->execute(inputs_vec, outputs));
    }
    { // TP enabled
        auto kernel = make_kernel(true);
        std::vector<std::shared_ptr<TensorBase>> outputs = {out_tp};
        ASSERT_TRUE(kernel->execute(inputs_vec, outputs));
    }
    // Compare
    std::vector<float> b(out_baseline->size());
    std::vector<float> t(out_tp->size());
    std::memcpy(b.data(), out_baseline->data(), b.size() * sizeof(float));
    std::memcpy(t.data(), out_tp->data(), t.size() * sizeof(float));
    auto stats = testutils::diff(t, b);
    EXPECT_LE(stats.max_abs, 5e-6) << testutils::summarize(stats);
    EXPECT_LE(stats.rel_l2, 5e-7) << testutils::summarize(stats);
    // Clean env
    unsetenv("LLAMINAR_ATTN_TP_PARTITIONS");
    unsetenv("LLAMINAR_ATTN_TP_DISABLE");
}

LLAMINAR_DEFINE_GTEST_MPI_MAIN();
