/**
 * @file Test__GDNTPOffsetRegression.cpp
 * @brief Regression tests for GDN `global_v_head_offset` under TP sharding
 *
 * Locks in the fix in `Qwen35Graph::buildGDNSubgraph` that was missing a
 * branch for the *expansion* GQA regime (`n_k_heads < n_v_heads`). In that
 * regime Q/K are replicated across TP ranks while V is sharded, and each
 * rank needs `global_v_head_offset = rank * n_v_heads_local` so the
 * deinterleave helper picks the correct K-heads for its V-head slice:
 *
 *     k_idx = (v_local + global_v_head_offset) % n_k_heads
 *
 * The bug left the offset at 0 on rank > 0 in expansion mode, causing
 * rank 1 to read the K-heads meant for rank 0. It manifested only when
 * `n_v_heads_local % n_k_heads != 0` (otherwise the period coincidentally
 * aligned across ranks). The 27B Qwen-3.5 TP=2 case (n_k=16, n_v_local=24)
 * hit exactly this.
 *
 * These tests validate the underlying invariant directly on
 * `GDNRecurrenceStage`:
 *
 *   1. `TPEquivalence_Expansion_Decode` — full run over N V-heads equals
 *      concat(rank 0 with N/2 V-heads offset=0, rank 1 with N/2 V-heads
 *      offset=N/2) on the decode code path.
 *   2. `TPEquivalence_Expansion_Prefill` — same, on the prefill (chunk
 *      forward) code path. This is the path that was broken for 27B.
 *   3. `Rank1_Offset0_Diverges_When_Period_Mismatch` — reproduces the
 *      original bug: running rank 1 with offset=0 in expansion mode
 *      when `n_v_local % n_k != 0` produces a DIFFERENT output from
 *      the correct split, so Qwen35Graph cannot silently regress to
 *      "offset doesn't matter here".
 *   4. `TPEquivalence_Identity_AllRanks` — sanity check for the
 *      identity regime (n_k == n_v_local, K replicated at full count).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        std::memset(t->mutable_data(), 0, t->numel() * sizeof(float));
        return t;
    }

    // Fill the [seq_len, qkv_dim] merged QKV buffer with deterministic random
    // values. Q occupies [0, n_k*d_k), K occupies [n_k*d_k, 2*n_k*d_k),
    // V occupies [2*n_k*d_k, 2*n_k*d_k + n_v*d_v).
    void fillQKVRandom(FP32Tensor &tensor,
                       int seq_len, int n_k, int n_v, int d_k, int d_v,
                       uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        float *data = tensor.mutable_data();
        const int qkv_dim = 2 * n_k * d_k + n_v * d_v;
        for (int i = 0; i < seq_len * qkv_dim; ++i)
            data[i] = dist(rng);
    }

    // Build a merged QKV tensor where the V-slice is restricted to the
    // local V-heads owned by `rank` (contiguous slice of the full V). Q/K
    // are replicated from the full buffer (they're identical across ranks).
    std::shared_ptr<FP32Tensor> shardedQKV(
        const FP32Tensor &full_qkv,
        int seq_len, int n_k, int n_v_full, int n_v_local,
        int d_k, int d_v, int rank)
    {
        const int q_dim = n_k * d_k;
        const int k_dim = n_k * d_k;
        const int v_full_dim = n_v_full * d_v;
        const int v_local_dim = n_v_local * d_v;
        const int full_stride = q_dim + k_dim + v_full_dim;
        const int local_stride = q_dim + k_dim + v_local_dim;

        auto out = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len),
                                static_cast<size_t>(local_stride)},
            DeviceId::cpu());
        const float *src = full_qkv.data();
        float *dst = out->mutable_data();
        for (int t = 0; t < seq_len; ++t)
        {
            const float *row_src = src + t * full_stride;
            float *row_dst = dst + t * local_stride;
            // Copy Q + K (full)
            std::memcpy(row_dst, row_src, (q_dim + k_dim) * sizeof(float));
            // Copy this rank's V slice
            const float *v_src = row_src + q_dim + k_dim + rank * v_local_dim;
            float *v_dst = row_dst + q_dim + k_dim;
            std::memcpy(v_dst, v_src, v_local_dim * sizeof(float));
        }
        return out;
    }

    // Slice a [seq_len, n_v_full] (or [n_v_full]) projection into the local
    // n_v_local range owned by `rank`.
    std::shared_ptr<FP32Tensor> sliceHeads1D(
        const FP32Tensor &full, int n_v_local, int rank)
    {
        const int start = rank * n_v_local;
        auto out = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_v_local)},
            DeviceId::cpu());
        std::memcpy(out->mutable_data(),
                    full.data() + start,
                    n_v_local * sizeof(float));
        return out;
    }

    std::shared_ptr<FP32Tensor> sliceHeads2D(
        const FP32Tensor &full, int seq_len, int n_v_full,
        int n_v_local, int rank)
    {
        const int start = rank * n_v_local;
        auto out = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len),
                                static_cast<size_t>(n_v_local)},
            DeviceId::cpu());
        const float *src = full.data();
        float *dst = out->mutable_data();
        for (int t = 0; t < seq_len; ++t)
        {
            std::memcpy(dst + t * n_v_local,
                        src + t * n_v_full + start,
                        n_v_local * sizeof(float));
        }
        return out;
    }

    struct RunResult
    {
        std::shared_ptr<FP32Tensor> output;
        std::vector<float> state;
    };

    // Execute GDNRecurrenceStage once with the given params. Returns the
    // output tensor plus a copy of the recurrence state after the run.
    RunResult runRecurrence(
        ITensor *qkv, ITensor *alpha, ITensor *beta,
        ITensor *A_log, ITensor *dt_bias,
        int seq_len, int n_v_heads, int n_k_heads, int d_k, int d_v,
        int global_v_head_offset, bool use_qk_l2norm)
    {
        static CPUGatedDeltaNet kernel;

        auto output = makeFP32({static_cast<size_t>(seq_len),
                                static_cast<size_t>(n_v_heads * d_v)});
        std::vector<float> state(
            static_cast<size_t>(n_v_heads) * d_k * d_v, 0.0f);

        GDNRecurrenceStage::Params p;
        p.kernel = &kernel;
        p.Q = qkv;
        p.K = qkv;
        p.V = qkv;
        p.alpha = alpha;
        p.beta = beta;
        p.A_log = A_log;
        p.dt_bias = dt_bias;
        p.output = output.get();
        p.recurrence_state = state.data();
        p.seq_len = seq_len;
        p.n_heads = n_v_heads;
        p.n_k_heads = n_k_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.chunk_size = 64;
        p.use_qk_l2norm = use_qk_l2norm;
        p.global_v_head_offset = global_v_head_offset;

        auto ctx = makeCPUContext();
        GDNRecurrenceStage stage(p);
        EXPECT_TRUE(stage.execute(ctx.get()));

        return {output, std::move(state)};
    }

    // Compare SD output [seq_len, n_v_full * d_v] against the concat of
    // rank 0 and rank 1 outputs, each [seq_len, n_v_local * d_v]. Verifies
    // per-head match within tolerance.
    ::testing::AssertionResult expectHeadsMatch(
        const FP32Tensor &sd_out,
        const FP32Tensor &rank_out,
        int seq_len, int n_v_full, int n_v_local,
        int d_v, int rank, float atol)
    {
        const int head_start = rank * n_v_local;
        const float *sd = sd_out.data();
        const float *lo = rank_out.data();
        const int sd_stride = n_v_full * d_v;
        const int lo_stride = n_v_local * d_v;
        for (int t = 0; t < seq_len; ++t)
        {
            for (int j = 0; j < n_v_local; ++j)
            {
                const int sd_h = head_start + j;
                for (int c = 0; c < d_v; ++c)
                {
                    const float a = sd[t * sd_stride + sd_h * d_v + c];
                    const float b = lo[t * lo_stride + j * d_v + c];
                    const float diff = std::abs(a - b);
                    if (diff > atol || !std::isfinite(a) || !std::isfinite(b))
                    {
                        return ::testing::AssertionFailure()
                               << "t=" << t << " head=" << sd_h
                               << " (local j=" << j << " rank=" << rank
                               << ") c=" << c << ": SD=" << a
                               << " TP=" << b << " diff=" << diff;
                    }
                }
            }
        }
        return ::testing::AssertionSuccess();
    }

    // Helper: average absolute difference between SD heads and a rank's
    // output. Used to check that the buggy path (offset=0 on rank 1)
    // produces noticeably different output.
    float meanAbsDiffHeads(
        const FP32Tensor &sd_out,
        const FP32Tensor &rank_out,
        int seq_len, int n_v_full, int n_v_local,
        int d_v, int rank)
    {
        const int head_start = rank * n_v_local;
        const float *sd = sd_out.data();
        const float *lo = rank_out.data();
        const int sd_stride = n_v_full * d_v;
        const int lo_stride = n_v_local * d_v;
        double acc = 0.0;
        long count = 0;
        for (int t = 0; t < seq_len; ++t)
        {
            for (int j = 0; j < n_v_local; ++j)
            {
                for (int c = 0; c < d_v; ++c)
                {
                    const float a = sd[t * sd_stride + (head_start + j) * d_v + c];
                    const float b = lo[t * lo_stride + j * d_v + c];
                    acc += std::abs(a - b);
                    ++count;
                }
            }
        }
        return static_cast<float>(acc / std::max<long>(count, 1));
    }
} // namespace

// ============================================================================
// TP Equivalence: Expansion regime (n_k < n_v), non-aligned period
// ============================================================================
//
// Setup: n_k_heads = 2 (replicated), n_v_heads_full = 6, TP = 2 →
//        n_v_heads_local = 3. Since 3 % 2 != 0, the modular K-head
//        pattern does NOT repeat across rank boundaries, which is the
//        exact condition that exposed the 27B bug.

TEST(Test__GDNTPOffsetRegression, TPEquivalence_Expansion_Decode)
{
    constexpr int n_k = 2;
    constexpr int n_v_full = 6;
    constexpr int n_v_local = 3;
    constexpr int d_k = 8;
    constexpr int d_v = 8;
    constexpr int seq_len = 1; // Decode path

    const int qkv_dim_full = 2 * n_k * d_k + n_v_full * d_v;

    // Build the reference merged QKV buffer with the full V dimension.
    auto full_qkv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, static_cast<size_t>(qkv_dim_full)},
        DeviceId::cpu());
    fillQKVRandom(*full_qkv, seq_len, n_k, n_v_full, d_k, d_v, /*seed=*/0xA17E'0FF5);

    // alpha/beta are n_v-sized [seq_len, n_v]
    auto alpha_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    auto beta_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    {
        std::mt19937 rng(0x3E7A);
        std::normal_distribution<float> dist(0.0f, 0.2f);
        for (int i = 0; i < seq_len * n_v_full; ++i)
        {
            alpha_full->mutable_data()[i] = dist(rng);
            beta_full->mutable_data()[i] = dist(rng);
        }
    }

    // A_log and dt_bias are [n_v_heads] — non-uniform so heads differ.
    auto A_log_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    auto dt_bias_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    for (int h = 0; h < n_v_full; ++h)
    {
        A_log_full->mutable_data()[h] = -1.0f - 0.05f * static_cast<float>(h);
        dt_bias_full->mutable_data()[h] = 0.03f * static_cast<float>(h);
    }

    // Single-device reference: full run, offset=0.
    auto sd = runRecurrence(
        full_qkv.get(), alpha_full.get(), beta_full.get(),
        A_log_full.get(), dt_bias_full.get(),
        seq_len, n_v_full, n_k, d_k, d_v,
        /*global_v_head_offset=*/0, /*use_qk_l2norm=*/true);

    // Per-rank runs: rank r owns V-heads [r*n_v_local, (r+1)*n_v_local),
    // offset = r*n_v_local, with proportional slices of alpha/beta/A_log/dt_bias.
    for (int rank = 0; rank < 2; ++rank)
    {
        auto rank_qkv = shardedQKV(*full_qkv, seq_len, n_k, n_v_full,
                                   n_v_local, d_k, d_v, rank);
        auto rank_alpha = sliceHeads2D(*alpha_full, seq_len, n_v_full,
                                       n_v_local, rank);
        auto rank_beta = sliceHeads2D(*beta_full, seq_len, n_v_full,
                                      n_v_local, rank);
        auto rank_alog = sliceHeads1D(*A_log_full, n_v_local, rank);
        auto rank_dt = sliceHeads1D(*dt_bias_full, n_v_local, rank);

        auto local = runRecurrence(
            rank_qkv.get(), rank_alpha.get(), rank_beta.get(),
            rank_alog.get(), rank_dt.get(),
            seq_len, n_v_local, n_k, d_k, d_v,
            /*global_v_head_offset=*/rank * n_v_local,
            /*use_qk_l2norm=*/true);

        EXPECT_TRUE(expectHeadsMatch(*sd.output, *local.output,
                                     seq_len, n_v_full, n_v_local,
                                     d_v, rank, /*atol=*/1e-5f))
            << "Rank " << rank << " (decode) does not match SD slice. "
            << "This indicates global_v_head_offset is not being honoured "
            << "in the expansion regime (n_k=" << n_k
            << " < n_v_local=" << n_v_local << ").";
    }
}

TEST(Test__GDNTPOffsetRegression, TPEquivalence_Expansion_Prefill)
{
    // Prefill path (seq_len > 1) uses chunk_forward, which is the code path
    // that was actually broken for 27B prefill parity. Same geometry as the
    // decode test.
    constexpr int n_k = 2;
    constexpr int n_v_full = 6;
    constexpr int n_v_local = 3;
    constexpr int d_k = 8;
    constexpr int d_v = 8;
    constexpr int seq_len = 5; // Prefill path, > 1 token

    const int qkv_dim_full = 2 * n_k * d_k + n_v_full * d_v;

    auto full_qkv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, static_cast<size_t>(qkv_dim_full)},
        DeviceId::cpu());
    fillQKVRandom(*full_qkv, seq_len, n_k, n_v_full, d_k, d_v, /*seed=*/0xBEEF'CAFE);

    auto alpha_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    auto beta_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    {
        std::mt19937 rng(0x1234);
        std::normal_distribution<float> dist(0.0f, 0.2f);
        for (int i = 0; i < seq_len * n_v_full; ++i)
        {
            alpha_full->mutable_data()[i] = dist(rng);
            beta_full->mutable_data()[i] = dist(rng);
        }
    }

    auto A_log_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    auto dt_bias_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    for (int h = 0; h < n_v_full; ++h)
    {
        A_log_full->mutable_data()[h] = -1.0f - 0.05f * static_cast<float>(h);
        dt_bias_full->mutable_data()[h] = 0.03f * static_cast<float>(h);
    }

    auto sd = runRecurrence(
        full_qkv.get(), alpha_full.get(), beta_full.get(),
        A_log_full.get(), dt_bias_full.get(),
        seq_len, n_v_full, n_k, d_k, d_v,
        /*global_v_head_offset=*/0, /*use_qk_l2norm=*/true);

    for (int rank = 0; rank < 2; ++rank)
    {
        auto rank_qkv = shardedQKV(*full_qkv, seq_len, n_k, n_v_full,
                                   n_v_local, d_k, d_v, rank);
        auto rank_alpha = sliceHeads2D(*alpha_full, seq_len, n_v_full,
                                       n_v_local, rank);
        auto rank_beta = sliceHeads2D(*beta_full, seq_len, n_v_full,
                                      n_v_local, rank);
        auto rank_alog = sliceHeads1D(*A_log_full, n_v_local, rank);
        auto rank_dt = sliceHeads1D(*dt_bias_full, n_v_local, rank);

        auto local = runRecurrence(
            rank_qkv.get(), rank_alpha.get(), rank_beta.get(),
            rank_alog.get(), rank_dt.get(),
            seq_len, n_v_local, n_k, d_k, d_v,
            /*global_v_head_offset=*/rank * n_v_local,
            /*use_qk_l2norm=*/true);

        EXPECT_TRUE(expectHeadsMatch(*sd.output, *local.output,
                                     seq_len, n_v_full, n_v_local,
                                     d_v, rank, /*atol=*/1e-4f))
            << "Rank " << rank << " (prefill/chunk_forward) does not match "
                                  "SD slice. Qwen35Graph may be dropping global_v_head_offset "
                                  "for the expansion regime.";
    }
}

// ============================================================================
// Original bug reproduction: offset=0 on rank 1 in expansion regime with
// non-aligned period produces the WRONG answer.
// ============================================================================

TEST(Test__GDNTPOffsetRegression, Rank1_Offset0_Diverges_When_Period_Mismatch)
{
    // Same geometry as TPEquivalence_Expansion_Prefill. With n_k=2 and
    // n_v_local=3, the modular K-head period does NOT align to rank
    // boundaries — so offset=0 on rank 1 MUST produce a different output
    // than offset=3. If this test ever passes with tiny diff, something
    // has regressed Qwen35Graph into not caring about the offset.
    constexpr int n_k = 2;
    constexpr int n_v_full = 6;
    constexpr int n_v_local = 3;
    constexpr int d_k = 8;
    constexpr int d_v = 8;
    constexpr int seq_len = 5;

    const int qkv_dim_full = 2 * n_k * d_k + n_v_full * d_v;

    auto full_qkv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, static_cast<size_t>(qkv_dim_full)},
        DeviceId::cpu());
    fillQKVRandom(*full_qkv, seq_len, n_k, n_v_full, d_k, d_v, /*seed=*/0xBEEF'CAFE);

    auto alpha_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    auto beta_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    {
        std::mt19937 rng(0x1234);
        std::normal_distribution<float> dist(0.0f, 0.2f);
        for (int i = 0; i < seq_len * n_v_full; ++i)
        {
            alpha_full->mutable_data()[i] = dist(rng);
            beta_full->mutable_data()[i] = dist(rng);
        }
    }

    auto A_log_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    auto dt_bias_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    for (int h = 0; h < n_v_full; ++h)
    {
        A_log_full->mutable_data()[h] = -1.0f - 0.05f * static_cast<float>(h);
        dt_bias_full->mutable_data()[h] = 0.03f * static_cast<float>(h);
    }

    auto sd = runRecurrence(
        full_qkv.get(), alpha_full.get(), beta_full.get(),
        A_log_full.get(), dt_bias_full.get(),
        seq_len, n_v_full, n_k, d_k, d_v,
        /*global_v_head_offset=*/0, /*use_qk_l2norm=*/true);

    constexpr int rank = 1;
    auto rank_qkv = shardedQKV(*full_qkv, seq_len, n_k, n_v_full,
                               n_v_local, d_k, d_v, rank);
    auto rank_alpha = sliceHeads2D(*alpha_full, seq_len, n_v_full,
                                   n_v_local, rank);
    auto rank_beta = sliceHeads2D(*beta_full, seq_len, n_v_full,
                                  n_v_local, rank);
    auto rank_alog = sliceHeads1D(*A_log_full, n_v_local, rank);
    auto rank_dt = sliceHeads1D(*dt_bias_full, n_v_local, rank);

    // Correct run: offset = rank * n_v_local = 3
    auto correct = runRecurrence(
        rank_qkv.get(), rank_alpha.get(), rank_beta.get(),
        rank_alog.get(), rank_dt.get(),
        seq_len, n_v_local, n_k, d_k, d_v,
        /*global_v_head_offset=*/rank * n_v_local,
        /*use_qk_l2norm=*/true);

    // Buggy run: offset = 0 (pre-fix behaviour on rank 1 in expansion mode)
    auto buggy = runRecurrence(
        rank_qkv.get(), rank_alpha.get(), rank_beta.get(),
        rank_alog.get(), rank_dt.get(),
        seq_len, n_v_local, n_k, d_k, d_v,
        /*global_v_head_offset=*/0,
        /*use_qk_l2norm=*/true);

    const float correct_err = meanAbsDiffHeads(
        *sd.output, *correct.output, seq_len, n_v_full, n_v_local, d_v, rank);
    const float buggy_err = meanAbsDiffHeads(
        *sd.output, *buggy.output, seq_len, n_v_full, n_v_local, d_v, rank);

    // Correct run matches SD closely.
    EXPECT_LT(correct_err, 1e-4f)
        << "Correct offset=" << (rank * n_v_local)
        << " should match SD, got mean abs diff " << correct_err;

    // Buggy run (offset=0) must be clearly worse than the correct run.
    // The exact magnitude depends on the random inputs, but with a
    // non-aligned period the error is large enough to be unmistakable
    // (several orders of magnitude above the correct-run residual).
    EXPECT_GT(buggy_err, correct_err * 100.0f)
        << "Rank 1 with offset=0 should diverge noticeably from SD in the "
           "expansion regime (n_k="
        << n_k << ", n_v_local=" << n_v_local
        << "). If this fails, the offset is being silently ignored or the "
           "period happens to align. correct_err="
        << correct_err << " buggy_err=" << buggy_err;
}

// ============================================================================
// Sanity: Identity regime (K replicated at full count, n_k == n_v_local)
// ============================================================================

TEST(Test__GDNTPOffsetRegression, TPEquivalence_Identity_AllRanks)
{
    // Identity regime: n_k_full = n_v_full, K replicated across ranks at
    // full count (not sharded). Each rank owns n_v_local = n_v_full / TP
    // contiguous V-heads. Using offset = rank * n_v_local here is still
    // correct (it rotates each rank's K-selection to match the SD heads
    // it owns).
    constexpr int n_k = 4; // == n_v_full, K replicated
    constexpr int n_v_full = 4;
    constexpr int n_v_local = 2; // TP=2
    constexpr int d_k = 8;
    constexpr int d_v = 8;
    constexpr int seq_len = 3;

    const int qkv_dim_full = 2 * n_k * d_k + n_v_full * d_v;

    auto full_qkv = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, static_cast<size_t>(qkv_dim_full)},
        DeviceId::cpu());
    fillQKVRandom(*full_qkv, seq_len, n_k, n_v_full, d_k, d_v, /*seed=*/0x1DEA);

    auto alpha_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    auto beta_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{seq_len, n_v_full}, DeviceId::cpu());
    {
        std::mt19937 rng(0x55AA);
        std::normal_distribution<float> dist(0.0f, 0.2f);
        for (int i = 0; i < seq_len * n_v_full; ++i)
        {
            alpha_full->mutable_data()[i] = dist(rng);
            beta_full->mutable_data()[i] = dist(rng);
        }
    }

    auto A_log_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    auto dt_bias_full = std::make_shared<FP32Tensor>(
        std::vector<size_t>{n_v_full}, DeviceId::cpu());
    for (int h = 0; h < n_v_full; ++h)
    {
        A_log_full->mutable_data()[h] = -1.0f - 0.1f * static_cast<float>(h);
        dt_bias_full->mutable_data()[h] = 0.02f * static_cast<float>(h);
    }

    auto sd = runRecurrence(
        full_qkv.get(), alpha_full.get(), beta_full.get(),
        A_log_full.get(), dt_bias_full.get(),
        seq_len, n_v_full, n_k, d_k, d_v,
        /*global_v_head_offset=*/0, /*use_qk_l2norm=*/true);

    for (int rank = 0; rank < 2; ++rank)
    {
        auto rank_qkv = shardedQKV(*full_qkv, seq_len, n_k, n_v_full,
                                   n_v_local, d_k, d_v, rank);
        auto rank_alpha = sliceHeads2D(*alpha_full, seq_len, n_v_full,
                                       n_v_local, rank);
        auto rank_beta = sliceHeads2D(*beta_full, seq_len, n_v_full,
                                      n_v_local, rank);
        auto rank_alog = sliceHeads1D(*A_log_full, n_v_local, rank);
        auto rank_dt = sliceHeads1D(*dt_bias_full, n_v_local, rank);

        auto local = runRecurrence(
            rank_qkv.get(), rank_alpha.get(), rank_beta.get(),
            rank_alog.get(), rank_dt.get(),
            seq_len, n_v_local, n_k, d_k, d_v,
            /*global_v_head_offset=*/rank * n_v_local,
            /*use_qk_l2norm=*/true);

        EXPECT_TRUE(expectHeadsMatch(*sd.output, *local.output,
                                     seq_len, n_v_full, n_v_local,
                                     d_v, rank, /*atol=*/1e-4f))
            << "Rank " << rank << " identity regime failed";
    }
}
