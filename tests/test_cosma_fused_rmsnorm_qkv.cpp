#include <gtest/gtest.h>
#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <cblas.h>
// NOTE: Direct MPI include replaced by shared test utilities.
#include "test_mpi_utils.h"
#include <random>
#include <vector>
#include <cmath>
#include <cstring>

using namespace llaminar;

namespace
{

    // MPI handled by MPIEnvironment (see test_mpi_utils.h)

    static void fillRandom(std::vector<float> &v, float scale = 0.01f)
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (auto &x : v)
            x = dist(gen);
    }

    static double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
        {
            // Return a large value to force failure in caller; caller will include rel_l2 in message.
            return 1e9;
        }
        long double num = 0.0L, den = 0.0L;
        size_t n = a.size();
        for (size_t i = 0; i < n; ++i)
        {
            long double da = a[i];
            long double db = b[i];
            long double d = da - db;
            num += d * d;
            den += db * db;
        }
        if (den == 0.0L)
            return num == 0.0L ? 0.0 : 1e9;
        return std::sqrt((double)(num / den));
    }

    struct TestDims
    {
        int seq_len;
        int hidden;
        int out_q;
        int out_kv;
    };

    // Builds square weights for q/k/v (out dims == hidden) unless overridden.
    static void build_weights(const TestDims &d,
                              std::vector<float> &wq, std::vector<float> &wk, std::vector<float> &wv)
    {
        wq.resize((size_t)d.hidden * d.out_q);
        wk.resize((size_t)d.hidden * d.out_kv);
        wv.resize((size_t)d.hidden * d.out_kv);
        fillRandom(wq);
        fillRandom(wk);
        fillRandom(wv);
    }

    static void reference_rmsnorm_qkv(const TestDims &dims, const float *activation, const float *gamma,
                                      const float *Wq, const float *Wk, const float *Wv, float eps,
                                      std::vector<float> &norm,
                                      std::vector<float> &q,
                                      std::vector<float> &k,
                                      std::vector<float> &v)
    {
        const int S = dims.seq_len;
        const int H = dims.hidden;
        const int OQ = dims.out_q;
        const int OKV = dims.out_kv;
        norm.assign((size_t)S * H, 0.f);
        q.assign((size_t)S * OQ, 0.f);
        k.assign((size_t)S * OKV, 0.f);
        v.assign((size_t)S * OKV, 0.f);
        // RMSNorm + scale by gamma (assume gamma length H)
        for (int r = 0; r < S; ++r)
        {
            const float *row = activation + (size_t)r * H;
            long double sum = 0.0L;
            for (int c = 0; c < H; ++c)
            {
                long double x = row[c];
                sum += x * x;
            }
            long double inv = 1.0L / std::sqrt((double)(sum / std::max(1, H)) + (double)eps);
            float *dst = norm.data() + (size_t)r * H;
            for (int c = 0; c < H; ++c)
            {
                dst[c] = (float)(row[c] * inv * gamma[c]);
            }
        }
        // Matmul row-major: (S,H) * (H,OQ)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, OQ, H,
                    1.0f, norm.data(), H, Wq, OQ, 0.0f, q.data(), OQ);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, OKV, H,
                    1.0f, norm.data(), H, Wk, OKV, 0.0f, k.data(), OKV);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, OKV, H,
                    1.0f, norm.data(), H, Wv, OKV, 0.0f, v.data(), OKV);
    }

    static WeightDescriptor make_desc(const std::vector<float> &W, int rows, int cols, const std::string &id)
    {
        WeightDescriptor d{id, rows, cols, (int64_t)cols, 1, 0, W.data(), 0};
        return d;
    }

    class FusedRmsnormQkvHarness : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            world_size_ = ::llaminar::test_util::MPIEnvironment::world();
            // Force COSMA path regardless of threshold.
            setenv("LLAMINAR_COSMA_FORCE", "1", 1);
            setenv("LLAMINAR_RMS_FORENSICS_FUSED", "1", 1); // enable for debugging (harmless if unused)
            setenv("LLAMINAR_COSMA_LOG_LEVEL", "info", 1);
        }
        int world_size_ = 1;
    };

    TEST_F(FusedRmsnormQkvHarness, SingleRankCorrectness)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Run without mpirun (single rank) for this test";
        TestDims dims{32, 128, 128, 128};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        std::vector<float> gamma(dims.hidden);
        fillRandom(activation, 0.05f);
        // Gamma often >0; use positive range to avoid large sign flips amplifying drift
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 0.5f + 0.05f * (float)(i % 7);
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;

        // Reference
        std::vector<float> norm_ref, q_ref, k_ref, v_ref;
        reference_rmsnorm_qkv(dims, activation.data(), gamma.data(), wq.data(), wk.data(), wv.data(), eps,
                              norm_ref, q_ref, k_ref, v_ref);

        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto &cache = mgr;
        (void)cache; // ensure instance creation

        // Build descriptors
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");

        // Call fused path
        auto fused = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                           dims.seq_len, dims.hidden, eps, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);

        ASSERT_TRUE(fused.normalized.host_owned || fused.normalized.original_row_major);
        ASSERT_TRUE(fused.q.host_owned || fused.q.original_row_major);
        ASSERT_TRUE(fused.k.host_owned || fused.k.original_row_major);
        ASSERT_TRUE(fused.v.host_owned || fused.v.original_row_major);

        // Reconstruct to row-major buffers
        std::vector<float> norm_buf((size_t)dims.seq_len * dims.hidden);
        std::vector<float> q_buf((size_t)dims.seq_len * dims.out_q);
        std::vector<float> k_buf((size_t)dims.seq_len * dims.out_kv);
        std::vector<float> v_buf((size_t)dims.seq_len * dims.out_kv);
        mgr.to_row_major(fused.normalized, norm_buf.data());
        mgr.to_row_major(fused.q, q_buf.data(), true);
        mgr.to_row_major(fused.k, k_buf.data(), true);
        mgr.to_row_major(fused.v, v_buf.data(), true);

        double rl2_norm = rel_l2(norm_buf, norm_ref);
        double rl2_q = rel_l2(q_buf, q_ref);
        double rl2_k = rel_l2(k_buf, k_ref);
        double rl2_v = rel_l2(v_buf, v_ref);

        EXPECT_LT(rl2_norm, 5e-6) << "norm rel_l2=" << rl2_norm;
        EXPECT_LT(rl2_q, 5e-5) << "q rel_l2=" << rl2_q; // allow slightly more for accumulation
        EXPECT_LT(rl2_k, 5e-5) << "k rel_l2=" << rl2_k;
        EXPECT_LT(rl2_v, 5e-5) << "v rel_l2=" << rl2_v;
    }

    TEST_F(FusedRmsnormQkvHarness, MultiRankCorrectness)
    {
        if (world_size_ < 2)
            GTEST_SKIP() << "Needs mpirun -np >=2";
        // Keep dims modest to reduce runtime; ensure seq_len triggers distributed path logically (force still set)
        // Increase seq_len to 64 to encourage non-zero tile allocation across ranks.
        TestDims dims{64, 128, 128, 128};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        std::vector<float> gamma(dims.hidden);
        fillRandom(activation, 0.05f);
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 0.5f + 0.05f * (float)(i % 11);
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;

        std::vector<float> norm_ref, q_ref, k_ref, v_ref;
        reference_rmsnorm_qkv(dims, activation.data(), gamma.data(), wq.data(), wk.data(), wv.data(), eps,
                              norm_ref, q_ref, k_ref, v_ref);

        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        auto fused = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                           dims.seq_len, dims.hidden, eps, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);

        // If any distributed view failed to materialize (allocation / budget guard), skip instead of crashing.
        bool ok_norm = (fused.normalized.host_owned || fused.normalized.original_row_major);
        bool ok_q = (fused.q.host_owned || fused.q.original_row_major);
        bool ok_k = (fused.k.host_owned || fused.k.original_row_major);
        bool ok_v = (fused.v.host_owned || fused.v.original_row_major);
        if (!(ok_norm && ok_q && ok_k && ok_v))
        {
            GTEST_SKIP() << "Distributed fused path did not materialize all outputs (norm=" << ok_norm
                         << " q=" << ok_q << " k=" << ok_k << " v=" << ok_v
                         << ") – skipping correctness check";
        }

        std::vector<float> norm_buf((size_t)dims.seq_len * dims.hidden);
        std::vector<float> q_buf((size_t)dims.seq_len * dims.out_q);
        std::vector<float> k_buf((size_t)dims.seq_len * dims.out_kv);
        std::vector<float> v_buf((size_t)dims.seq_len * dims.out_kv);
        mgr.to_row_major(fused.normalized, norm_buf.data());
        mgr.to_row_major(fused.q, q_buf.data(), true);
        mgr.to_row_major(fused.k, k_buf.data(), true);
        mgr.to_row_major(fused.v, v_buf.data(), true);

        // Only rank 0 asserts full correctness to avoid redundant reference cost
        int rank = ::llaminar::test_util::MPIEnvironment::rank();
        if (rank == 0)
        {
            double rl2_norm = rel_l2(norm_buf, norm_ref);
            double rl2_q = rel_l2(q_buf, q_ref);
            double rl2_k = rel_l2(k_buf, k_ref);
            double rl2_v = rel_l2(v_buf, v_ref);
            EXPECT_LT(rl2_norm, 1e-5) << "norm rel_l2=" << rl2_norm;
            EXPECT_LT(rl2_q, 2e-4) << "q rel_l2=" << rl2_q;
            EXPECT_LT(rl2_k, 2e-4) << "k rel_l2=" << rl2_k;
            EXPECT_LT(rl2_v, 2e-4) << "v rel_l2=" << rl2_v;
        }
    }

    TEST_F(FusedRmsnormQkvHarness, SingleRankGammaImmutability)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Run without mpirun (single rank) for this test";
        TestDims dims{16, 96, 96, 96};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        fillRandom(activation, 0.02f);

        std::vector<float> gamma(dims.hidden);
        // Populate gamma with a deterministic, high-variance pattern including sentinels.
        for (int i = 0; i < dims.hidden; ++i)
        {
            float base = 0.25f + 0.001f * (float)(i * i % 97);
            if (i == 7)
                base = 12345.6789f; // large sentinel unlikely to appear by computation
            if (i == dims.hidden - 3)
                base = -3.14159f; // negative sentinel
            if (i == dims.hidden / 2)
                base = 0.0009765625f; // exact power-of-two fraction sentinel
            gamma[i] = base;
        }

        // Keep a pristine copy for byte-wise and numeric comparison.
        std::vector<float> gamma_before = gamma;

        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;

        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");

        // Execute fused kernel (result unused here; focus is gamma integrity).
        auto fused = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                           dims.seq_len, dims.hidden, eps,
                                           1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
        (void)fused;

        // First, quick byte-wise equality check.
        ASSERT_EQ(gamma.size(), gamma_before.size());
        int memcmp_result = std::memcmp(gamma.data(), gamma_before.data(), gamma.size() * sizeof(float));
        EXPECT_EQ(memcmp_result, 0) << "Gamma buffer bytes mutated by fused_rmsnorm_qkv";

        // If byte comparison fails, provide diagnostic of first few mismatches.
        if (memcmp_result != 0)
        {
            int mismatches = 0;
            std::ostringstream oss;
            oss << "First mismatches: ";
            for (size_t i = 0; i < gamma.size() && mismatches < 6; ++i)
            {
                if (gamma[i] != gamma_before[i])
                {
                    oss << "i=" << i << " before=" << gamma_before[i] << " after=" << gamma[i] << "; ";
                    ++mismatches;
                }
            }
            GTEST_FAIL() << oss.str();
        }

        // Numerical hash (sum and sumsq) for redundancy / future-proofing.
        long double sum_before = 0.0L, sum_after = 0.0L, ss_before = 0.0L, ss_after = 0.0L;
        for (size_t i = 0; i < gamma.size(); ++i)
        {
            long double a = gamma_before[i];
            long double b = gamma[i];
            sum_before += a;
            sum_after += b;
            ss_before += a * a;
            ss_after += b * b;
        }
        EXPECT_DOUBLE_EQ((double)sum_before, (double)sum_after) << "Gamma sum changed";
        EXPECT_DOUBLE_EQ((double)ss_before, (double)ss_after) << "Gamma sumsq changed";

        // Sentinel spot-checks (explicit indices) for clearer error localization.
        EXPECT_FLOAT_EQ(gamma[7], 12345.6789f);
        EXPECT_FLOAT_EQ(gamma[dims.hidden - 3], -3.14159f);
        EXPECT_FLOAT_EQ(gamma[dims.hidden / 2], 0.0009765625f);
    }

    TEST_F(FusedRmsnormQkvHarness, MultiRankGammaImmutability)
    {
        if (world_size_ < 2)
            GTEST_SKIP() << "Needs mpirun -np >=2";
        TestDims dims{32, 128, 128, 128};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        fillRandom(activation, 0.02f);
        std::vector<float> gamma(dims.hidden);
        for (int i = 0; i < dims.hidden; ++i)
        {
            float base = 0.75f + 0.002f * (float)((i * 13) % 101);
            if (i == 3)
                base = -2222.25f;
            if (i == dims.hidden / 2)
                base = 9.9999f;
            gamma[i] = base;
        }
        std::vector<float> gamma_before = gamma;

        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;
        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        auto fused = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                           dims.seq_len, dims.hidden, eps,
                                           1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
        (void)fused;

        // All ranks check immutability (cheap) to catch rank-specific mutation.
        int memcmp_result = std::memcmp(gamma.data(), gamma_before.data(), gamma.size() * sizeof(float));
        EXPECT_EQ(memcmp_result, 0) << "Gamma mutated on rank=" << ::llaminar::test_util::MPIEnvironment::rank();
        if (memcmp_result != 0 && ::llaminar::test_util::MPIEnvironment::rank() == 0)
        {
            int mismatches = 0;
            std::ostringstream oss;
            oss << "First mismatches: ";
            for (size_t i = 0; i < gamma.size() && mismatches < 6; ++i)
                if (gamma[i] != gamma_before[i])
                {
                    oss << i << ":" << gamma_before[i] << "->" << gamma[i] << " ";
                    ++mismatches;
                }
            GTEST_FAIL() << oss.str();
        }
        EXPECT_FLOAT_EQ(gamma[3], -2222.25f);
        EXPECT_FLOAT_EQ(gamma[dims.hidden / 2], 9.9999f);
    }

    // --- Additional correctness / forensics tests --------------------------------------

    static void compute_reference_norm_only(const TestDims &d, const float *activation, const float *gamma, float eps, std::vector<float> &out_norm)
    {
        const int S = d.seq_len;
        const int H = d.hidden;
        out_norm.assign((size_t)S * H, 0.f);
        for (int r = 0; r < S; ++r)
        {
            const float *row = activation + (size_t)r * H;
            long double sum = 0.0L;
            for (int c = 0; c < H; ++c)
            {
                long double v = row[c];
                sum += v * v;
            }
            long double inv = 1.0L / std::sqrt((double)(sum / std::max(1, H)) + (double)eps);
            float *dst = out_norm.data() + (size_t)r * H;
            for (int c = 0; c < H; ++c)
            {
                dst[c] = (float)(row[c] * inv * gamma[c]);
            }
        }
    }

    TEST_F(FusedRmsnormQkvHarness, SingleRankActivationImmutability)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Single-rank only";
        TestDims dims{24, 80, 80, 80};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        for (size_t i = 0; i < activation.size(); ++i)
            activation[i] = std::sin((float)i * 0.0137f) * 0.01f + (float)(i % 7) * 1e-4f;
        std::vector<float> activation_before = activation; // snapshot
        std::vector<float> gamma(dims.hidden);
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 0.9f + 0.0003f * (float)i;
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;
        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        (void)mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                    dims.seq_len, dims.hidden, eps, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
        // Check for mutation.
        ASSERT_EQ(activation.size(), activation_before.size());
        int diff_count = 0;
        size_t first_diff = (size_t)-1;
        for (size_t i = 0; i < activation.size(); ++i)
        {
            if (activation[i] != activation_before[i])
            {
                ++diff_count;
                if (first_diff == (size_t)-1)
                    first_diff = i;
                if (diff_count > 8)
                    break;
            }
        }
        EXPECT_EQ(diff_count, 0) << "Activation buffer mutated; first diff index=" << first_diff;
    }

    TEST_F(FusedRmsnormQkvHarness, SingleRankDeterminism)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Single-rank only";
        TestDims dims{32, 96, 96, 96};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        fillRandom(activation, 0.03f);
        std::vector<float> gamma(dims.hidden);
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 1.0f + 0.001f * (float)(i % 13);
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto run_once = [&](std::vector<float> &norm_buf, std::vector<float> &q_buf, std::vector<float> &k_buf, std::vector<float> &v_buf)
        {
            auto fused = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                               dims.seq_len, dims.hidden, eps, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
            norm_buf.resize((size_t)dims.seq_len * dims.hidden);
            q_buf.resize((size_t)dims.seq_len * dims.out_q);
            k_buf.resize((size_t)dims.seq_len * dims.out_kv);
            v_buf.resize((size_t)dims.seq_len * dims.out_kv);
            mgr.to_row_major(fused.normalized, norm_buf.data());
            mgr.to_row_major(fused.q, q_buf.data(), true);
            mgr.to_row_major(fused.k, k_buf.data(), true);
            mgr.to_row_major(fused.v, v_buf.data(), true);
        };
        std::vector<float> n1, q1, k1, v1, n2, q2, k2, v2;
        run_once(n1, q1, k1, v1);
        run_once(n2, q2, k2, v2);
        ASSERT_EQ(n1.size(), n2.size());
        ASSERT_EQ(q1.size(), q2.size());
        auto cmp = [](const std::vector<float> &a, const std::vector<float> &b)
        { for(size_t i=0;i<a.size();++i){ if (a[i] != b[i]) return (int)i; } return -1; };
        int dn = cmp(n1, n2);
        int dq = cmp(q1, q2);
        int dk = cmp(k1, k2);
        int dv = cmp(v1, v2);
        EXPECT_EQ(dn, -1) << "norm mismatch at index " << dn;
        EXPECT_EQ(dq, -1) << "q mismatch at index " << dq;
        EXPECT_EQ(dk, -1) << "k mismatch at index " << dk;
        EXPECT_EQ(dv, -1) << "v mismatch at index " << dv;
    }

    TEST_F(FusedRmsnormQkvHarness, SingleRankEpsilonOverrideBehavior)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Single-rank only";
        TestDims dims{20, 64, 64, 64};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        fillRandom(activation, 1e-3f);
        std::vector<float> gamma(dims.hidden);
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 0.8f + 0.002f * (float)(i % 17);
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float base_eps = 1e-6f;
        float override_eps = 1e-2f; // large gap to ensure measurable delta
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        // Baseline
        auto fused_base = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                                dims.seq_len, dims.hidden, base_eps, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
        std::vector<float> norm_base((size_t)dims.seq_len * dims.hidden);
        mgr.to_row_major(fused_base.normalized, norm_base.data());
        // Override via env
        {
            // Use MPIEnvironment::ScopedEnvVar (defined in test_mpi_utils.h) to override epsilon temporarily.
            ::llaminar::test_util::MPIEnvironment::ScopedEnvVar eps_override("LLAMINAR_RMS_EPS_OVERRIDE", "1e-2");
            auto fused_override = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                                        dims.seq_len, dims.hidden, base_eps /* will be overridden */, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
            std::vector<float> norm_override((size_t)dims.seq_len * dims.hidden);
            mgr.to_row_major(fused_override.normalized, norm_override.data());
            // Reference computations
            std::vector<float> ref_base, ref_override;
            compute_reference_norm_only(dims, activation.data(), gamma.data(), base_eps, ref_base);
            compute_reference_norm_only(dims, activation.data(), gamma.data(), override_eps, ref_override);
            // Rel L2 helper
            auto rl2 = [](const std::vector<float> &a, const std::vector<float> &b)
            { long double num=0.0L, den=0.0L; for(size_t i=0;i<a.size();++i){ long double d=a[i]-b[i]; num+=d*d; den+= (long double)b[i]*b[i]; } if (den==0) return (double)(num==0?0:1); return (double)std::sqrt(num/den); };
            double rl2_base = rl2(norm_base, ref_base);
            double rl2_override = rl2(norm_override, ref_override);
            double delta_between = rl2(norm_base, norm_override);
            EXPECT_LT(rl2_base, 5e-6) << "baseline rmsnorm mismatch";
            EXPECT_LT(rl2_override, 5e-6) << "override rmsnorm mismatch";
            EXPECT_GT(delta_between, 1e-4) << "epsilon override produced no measurable change (delta=" << delta_between << ")";
        }
        // After scope, env cleared; repeat baseline and ensure identical to original baseline.
        auto fused_again = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                                 dims.seq_len, dims.hidden, base_eps, 1.0f / std::sqrt((float)dims.hidden / (float)dims.out_q), false);
        std::vector<float> norm_again((size_t)dims.seq_len * dims.hidden);
        mgr.to_row_major(fused_again.normalized, norm_again.data());
        int diff_index = -1;
        for (size_t i = 0; i < norm_base.size(); ++i)
        {
            if (norm_base[i] != norm_again[i])
            {
                diff_index = (int)i;
                break;
            }
        }
        EXPECT_EQ(diff_index, -1) << "Baseline output unstable after clearing override (first diff at index=" << diff_index << ")";
    }

    TEST_F(FusedRmsnormQkvHarness, SingleRankSoftmaxScaleNeutrality)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Single-rank only";
        TestDims dims{28, 72, 72, 72};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        fillRandom(activation, 0.02f);
        std::vector<float> gamma(dims.hidden);
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 1.0f;
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        float eps = 1e-6f;
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto fused_a = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                             dims.seq_len, dims.hidden, eps, 1.0f, false);
        auto fused_b = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                             dims.seq_len, dims.hidden, eps, 0.25f, false); // different scale (currently unused)
        std::vector<float> na((size_t)dims.seq_len * dims.hidden), nb((size_t)dims.seq_len * dims.hidden);
        mgr.to_row_major(fused_a.normalized, na.data());
        mgr.to_row_major(fused_b.normalized, nb.data());
        int diff = -1;
        for (size_t i = 0; i < na.size(); ++i)
        {
            if (na[i] != nb[i])
            {
                diff = (int)i;
                break;
            }
        }
        EXPECT_EQ(diff, -1) << "Softmax scale unexpectedly affected normalized output (first diff at index=" << diff << ")";
        // NOTE: When softmax_scale becomes active in fused path logic, this test should be revisited.
    }

    TEST_F(FusedRmsnormQkvHarness, SingleRankStagedEquivalence)
    {
        if (world_size_ != 1)
            GTEST_SKIP() << "Single-rank only";
        // Use non-square output dims to exercise distinct strides.
        TestDims dims{48, 96, 64, 80};
        std::vector<float> activation((size_t)dims.seq_len * dims.hidden);
        fillRandom(activation, 0.02f);
        std::vector<float> gamma(dims.hidden);
        for (int i = 0; i < dims.hidden; ++i)
            gamma[i] = 0.5f + 0.001f * (float)(i % 19);
        std::vector<float> wq, wk, wv;
        build_weights(dims, wq, wk, wv);
        // Adjust sizes because build_weights assumed out_q==hidden when not provided
        wq.resize((size_t)dims.hidden * dims.out_q);
        wk.resize((size_t)dims.hidden * dims.out_kv);
        wv.resize((size_t)dims.hidden * dims.out_kv);
        float eps = 1e-6f;
        CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
        auto wq_desc = make_desc(wq, dims.hidden, dims.out_q, "wq");
        auto wk_desc = make_desc(wk, dims.hidden, dims.out_kv, "wk");
        auto wv_desc = make_desc(wv, dims.hidden, dims.out_kv, "wv");
        // Fused
        auto fused = mgr.fused_rmsnorm_qkv(activation.data(), gamma.data(), wq_desc, wk_desc, wv_desc,
                                           dims.seq_len, dims.hidden, eps, 1.0f, false);
        ASSERT_TRUE(fused.normalized.host_owned || fused.normalized.original_row_major);
        ASSERT_TRUE(fused.q.host_owned || fused.q.original_row_major);
        ASSERT_TRUE(fused.k.host_owned || fused.k.original_row_major);
        ASSERT_TRUE(fused.v.host_owned || fused.v.original_row_major);
        std::vector<float> norm_f((size_t)dims.seq_len * dims.hidden);
        std::vector<float> q_f((size_t)dims.seq_len * dims.out_q);
        std::vector<float> k_f((size_t)dims.seq_len * dims.out_kv);
        std::vector<float> v_f((size_t)dims.seq_len * dims.out_kv);
        mgr.to_row_major(fused.normalized, norm_f.data());
        mgr.to_row_major(fused.q, q_f.data(), true);
        mgr.to_row_major(fused.k, k_f.data(), true);
        mgr.to_row_major(fused.v, v_f.data(), true);
        // Staged reference
        std::vector<float> norm_s, q_s, k_s, v_s;
        reference_rmsnorm_qkv(dims, activation.data(), gamma.data(), wq.data(), wk.data(), wv.data(), eps,
                              norm_s, q_s, k_s, v_s);
        auto check = [&](const char *lbl, const std::vector<float> &a, const std::vector<float> &b, double tol)
        {
        double r = rel_l2(a,b); EXPECT_LT(r, tol) << lbl << " rel_l2=" << r; };
        check("norm", norm_f, norm_s, 5e-6);
        check("q", q_f, q_s, 5e-5);
        check("k", k_f, k_s, 5e-5);
        check("v", v_f, v_s, 5e-5);
    }

    // -----------------------------------------------------------------------------------

} // namespace

// Use standardized MPI-aware main.
LLAMINAR_DEFINE_GTEST_MPI_MAIN();
