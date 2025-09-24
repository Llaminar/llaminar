#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <thread>
#include <atomic>
#include <chrono>
#include <execinfo.h>
#include <csignal>
#include <unistd.h>

using namespace llaminar;

static void fill_rand(std::vector<float> &v, int seed_off = 0)
{
    std::mt19937 gen(1337 + seed_off + (int)v.size());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : v)
        x = dist(gen);
}

struct MPIFinalizerElementwiseMulti
{
    ~MPIFinalizerElementwiseMulti()
    {
        int init = 0, fin = 0;
        MPI_Initialized(&init);
        if (init)
        {
            MPI_Finalized(&fin);
            if (!fin)
                MPI_Finalize();
        }
    }
};
static MPIFinalizerElementwiseMulti _mpi_finalizer;

TEST(CosmaInLayoutElementwiseMulti, DistributedRMSNormSwiGLUConsistency)
{
    // ---------------- Watchdog ----------------
    static std::atomic<bool> done{false};
    done.store(false);
    int internal_timeout_ms = 30000; // < ctest timeout (60)
    if (const char *env = std::getenv("LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS"))
    {
        int v = std::atoi(env);
        if (v > 0)
            internal_timeout_ms = v;
        else if (v == 0)
            internal_timeout_ms = -1;
    }
    auto start_tp = std::chrono::steady_clock::now();
    auto stack_dump = [](int rank)
    {
        void *frames[64];
        int n = ::backtrace(frames, 64);
        char **syms = ::backtrace_symbols(frames, n);
        fprintf(stderr, "[WATCHDOG][rank %d] === STACK TRACE (%d frames) ===\n", rank, n);
        for (int i = 0; i < n; i++)
            fprintf(stderr, "[WATCHDOG][rank %d] %s\n", rank, syms[i]);
        if (syms)
            free(syms);
        fflush(stderr);
    };
    std::thread watchdog([&]
                         {
        if (internal_timeout_ms < 0) return; 
        while(!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            auto now = std::chrono::steady_clock::now();
            auto el = std::chrono::duration_cast<std::chrono::milliseconds>(now-start_tp).count();
            if (el > internal_timeout_ms && !done.load()) {
                int r=0,w=1; MPI_Initialized(nullptr); MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&w);
                fprintf(stderr,"[WATCHDOG][rank %d/%d] Elapsed %lld ms > %d ms. Aborting.\n", r,w,(long long)el, internal_timeout_ms); fflush(stderr);
                stack_dump(r); ::raise(SIGABRT);
            }
        } });
    struct WatchdogJoin
    {
        std::thread &t;
        ~WatchdogJoin()
        {
            done.store(true);
            if (t.joinable())
                t.join();
        }
    } _wd_join{watchdog};
    // ------------------------------------------

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP() << "Need >=2 ranks";

    // Force disabling fast-path skip so COSMA allocation happens (we need distributed layout)
    setenv("LLAMINAR_COSMA_FAST_PATH_THRESHOLD", "1", 1);
    setenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT", "1", 1);
    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();

    const std::pair<int, int> shape_cases[] = {
        {64, 96}, // original shape where rank ownership may be uneven
        {96, 128} // additional coverage to ensure both ranks own tiles
    };

    for (const auto &[seq_len, hidden] : shape_cases)
    {
        SCOPED_TRACE(testing::Message() << "seq_len=" << seq_len << ", hidden=" << hidden);
        const int elements = seq_len * hidden;

        std::vector<float> act(elements), gamma(hidden), gate(elements), up(elements);
        fill_rand(act, 0);
        fill_rand(gamma, 1);
        fill_rand(gate, 2);
        fill_rand(up, 3);
        for (auto &g : gamma)
            g = std::fabs(g) + 0.5f; // positive scaling

        const auto &strat = mgr.strategy_for(seq_len, hidden, hidden);

        auto act_view = mgr.convert_activation_in_with_strategy(act.data(), seq_len, hidden, strat);
        std::vector<float> rms_out(elements, 0.f);
        auto out_view = mgr.convert_activation_in_with_strategy(rms_out.data(), seq_len, hidden, strat);
        ASSERT_TRUE(act_view.mat) << "Activation view lacked COSMA matrix (skip path triggered)";
        ASSERT_TRUE(out_view.mat) << "Output view lacked COSMA matrix (skip path triggered)";

        ASSERT_TRUE(mgr.rmsnorm_in_layout(act_view, out_view, gamma.data(), seq_len, hidden, 1e-5f));

        std::vector<float> gathered_rms(elements, 0.f);
        mgr.to_row_major(out_view, gathered_rms.data());

        std::vector<float> ref_rms(elements, 0.f);
        if (rank == 0)
        {
            for (int r = 0; r < seq_len; ++r)
            {
                const float *row = act.data() + (size_t)r * hidden;
                double sum = 0.0;
                for (int c = 0; c < hidden; ++c)
                {
                    double v = row[c];
                    sum += v * v;
                }
                double inv = 1.0 / std::sqrt(sum / hidden + 1e-5);
                for (int c = 0; c < hidden; ++c)
                    ref_rms[r * hidden + c] = float(row[c] * inv * gamma[c]);
            }
        }
        MPI_Bcast(ref_rms.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < elements; ++i)
        {
            double d = (double)gathered_rms[i] - ref_rms[i];
            num += d * d;
            den += (double)ref_rms[i] * ref_rms[i];
        }
        double rel_l2_rms = std::sqrt(num / (den + 1e-30));
        if (rank == 0)
        {
            LOG_INFO("[ElementwiseMulti] RMSNorm rel_l2=" << rel_l2_rms);
        }
        EXPECT_LT(rel_l2_rms, 1e-4) << "RMSNorm rel L2 too large";

        auto gate_view = mgr.convert_activation_in_with_strategy(gate.data(), seq_len, hidden, strat);
        auto up_view = mgr.convert_activation_in_with_strategy(up.data(), seq_len, hidden, strat);
        std::vector<float> sw_out(elements, 0.f);
        auto out_swiglu = mgr.convert_activation_in_with_strategy(sw_out.data(), seq_len, hidden, strat);

        ASSERT_TRUE(mgr.swiglu_in_layout(gate_view, up_view, out_swiglu, seq_len, hidden));
        std::vector<float> gathered_sw(elements, 0.f);
        mgr.to_row_major(out_swiglu, gathered_sw.data());

        std::vector<float> ref_sw(elements, 0.f);
        if (rank == 0)
        {
            auto silu = [](float x)
            { return x / (1.0f + std::exp(-x)); };
            for (int r = 0; r < seq_len; ++r)
                for (int c = 0; c < hidden; ++c)
                    ref_sw[r * hidden + c] = silu(up[r * hidden + c]) * gate[r * hidden + c];
        }
        MPI_Bcast(ref_sw.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

        num = 0.0;
        den = 0.0;
        for (int i = 0; i < elements; ++i)
        {
            double d = (double)gathered_sw[i] - ref_sw[i];
            num += d * d;
            den += (double)ref_sw[i] * ref_sw[i];
        }
        double rel_l2_sw = std::sqrt(num / (den + 1e-30));
        if (rank == 0)
        {
            LOG_INFO("[ElementwiseMulti] SwiGLU rel_l2=" << rel_l2_sw);
        }
        EXPECT_LT(rel_l2_sw, 1e-4) << "SwiGLU rel L2 too large";
    }

    done.store(true);
}

TEST(CosmaInLayoutElementwiseMulti, DistributedSoftmax)
{
    static std::atomic<bool> done_softmax{false};
    done_softmax.store(false);
    int timeout_ms = 30000;
    if (const char *env = std::getenv("LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS"))
    {
        int v = std::atoi(env);
        if (v > 0)
            timeout_ms = v;
        else if (v == 0)
            timeout_ms = -1;
    }
    auto start = std::chrono::steady_clock::now();
    std::thread watchdog([&]
                         {
                             if (timeout_ms < 0)
                                 return;
                             while (!done_softmax.load())
                             {
                                 std::this_thread::sleep_for(std::chrono::milliseconds(250));
                                 auto now = std::chrono::steady_clock::now();
                                 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                                 if (elapsed > timeout_ms && !done_softmax.load())
                                 {
                                     int r = 0, w = 1;
                                     MPI_Comm_rank(MPI_COMM_WORLD, &r);
                                     MPI_Comm_size(MPI_COMM_WORLD, &w);
                                     fprintf(stderr, "[WATCHDOG][softmax][rank %d/%d] Timeout after %lld ms\n", r, w, (long long)elapsed);
                                     ::raise(SIGABRT);
                                 }
                             } });
    struct WatchdogJoin
    {
        std::thread &t;
        ~WatchdogJoin()
        {
            done_softmax.store(true);
            if (t.joinable())
                t.join();
        }
    } _join{watchdog};

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP() << "Need >=2 ranks";

    setenv("LLAMINAR_COSMA_FAST_PATH_THRESHOLD", "1", 1);
    setenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT", "1", 1);
    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    mgr.set_force_cosma(true);

    const int seq_cases[] = {48, 64};
    for (int seq_len : seq_cases)
    {
        const int cols = seq_len;
        const int elements = seq_len * cols;
        std::vector<float> scores(elements);
        fill_rand(scores, seq_len);
        const auto &strat = mgr.strategy_for(seq_len, cols, cols);
        auto scores_view = mgr.convert_activation_in_with_strategy(scores.data(), seq_len, cols, strat);
        std::vector<float> softmax_out(elements, 0.f);
        auto out_view = mgr.convert_activation_in_with_strategy(softmax_out.data(), seq_len, cols, strat);

        ASSERT_TRUE(scores_view.mat) << "Distributed softmax requires COSMA layout";
        ASSERT_TRUE(out_view.mat);
        ASSERT_TRUE(mgr.softmax_in_layout(scores_view, out_view, seq_len, cols, 1.0f));

        std::vector<float> gathered(elements, 0.f);
        mgr.to_row_major(out_view, gathered.data());

        std::vector<float> ref(elements, 0.f);
        if (rank == 0)
        {
            for (int r = 0; r < seq_len; ++r)
            {
                const float *row = scores.data() + (size_t)r * cols;
                float row_max = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < cols; ++c)
                    row_max = std::max(row_max, row[c]);
                float denom = 0.f;
                for (int c = 0; c < cols; ++c)
                {
                    float val = std::exp(row[c] - row_max);
                    ref[(size_t)r * cols + c] = val;
                    denom += val;
                }
                float inv = denom > 0.f ? 1.f / denom : 1.f;
                for (int c = 0; c < cols; ++c)
                    ref[(size_t)r * cols + c] *= inv;
            }
        }
        MPI_Bcast(ref.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < elements; ++i)
        {
            double diff = (double)gathered[i] - (double)ref[i];
            num += diff * diff;
            den += (double)ref[i] * (double)ref[i];
        }
        double rel_l2 = std::sqrt(num / (den + 1e-30));
        if (rank == 0)
            LOG_INFO("[ElementwiseMulti] Softmax rel_l2=" << rel_l2 << " seq_len=" << seq_len);
        EXPECT_LT(rel_l2, 5e-4) << "Softmax rel L2 too large";
    }

    done_softmax.store(true);
}

TEST(CosmaInLayoutElementwiseMulti, FusedRmsnormQkv)
{
    static std::atomic<bool> done_fused{false};
    done_fused.store(false);
    int timeout_ms = 30000;
    if (const char *env = std::getenv("LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS"))
    {
        int v = std::atoi(env);
        if (v > 0)
            timeout_ms = v;
        else if (v == 0)
            timeout_ms = -1;
    }
    auto start = std::chrono::steady_clock::now();
    std::thread watchdog([&]
                         {
                             if (timeout_ms < 0)
                                 return;
                             while (!done_fused.load())
                             {
                                 std::this_thread::sleep_for(std::chrono::milliseconds(250));
                                 auto now = std::chrono::steady_clock::now();
                                 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
                                 if (elapsed > timeout_ms && !done_fused.load())
                                 {
                                     int r = 0, w = 1;
                                     MPI_Comm_rank(MPI_COMM_WORLD, &r);
                                     MPI_Comm_size(MPI_COMM_WORLD, &w);
                                     fprintf(stderr, "[WATCHDOG][fused][rank %d/%d] Timeout after %lld ms\n", r, w, (long long)elapsed);
                                     ::raise(SIGABRT);
                                 }
                             } });
    struct WatchdogJoin
    {
        std::thread &t;
        ~WatchdogJoin()
        {
            done_fused.store(true);
            if (t.joinable())
                t.join();
        }
    } _join{watchdog};

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world < 2)
        GTEST_SKIP() << "Need >=2 ranks";

    setenv("LLAMINAR_COSMA_FAST_PATH_THRESHOLD", "1", 1);
    setenv("LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT", "1", 1);
    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    mgr.set_force_cosma(true);

    const int seq_len = 48;
    const int hidden = 64;
    const float eps = 1e-5f;
    const int elements = seq_len * hidden;

    std::vector<float> act(elements);
    std::vector<float> gamma(hidden);
    std::vector<float> wq(hidden * hidden);
    std::vector<float> wk(hidden * hidden);
    std::vector<float> wv(hidden * hidden);
    fill_rand(act, 11);
    fill_rand(gamma, 12);
    fill_rand(wq, 13);
    fill_rand(wk, 14);
    fill_rand(wv, 15);
    for (auto &g : gamma)
        g = std::fabs(g) + 0.5f;

    WeightDescriptor desc_q{"q", hidden, hidden, hidden, 1, 0, wq.data(), 0};
    WeightDescriptor desc_k{"k", hidden, hidden, hidden, 1, 0, wk.data(), 0};
    WeightDescriptor desc_v{"v", hidden, hidden, hidden, 1, 0, wv.data(), 0};

    for (bool overlap : {false, true})
    {
        if (overlap)
            setenv("LLAMINAR_COSMA_OVERLAP_STREAM", "1", 1);
        else
            unsetenv("LLAMINAR_COSMA_OVERLAP_STREAM");

        auto fused = mgr.fused_rmsnorm_qkv(act.data(), gamma.data(), desc_q, desc_k, desc_v, seq_len, hidden, eps);
        ASSERT_TRUE(fused.normalized.mat) << "Normalized view missing distributed layout";
        ASSERT_TRUE(fused.q.mat && fused.k.mat && fused.v.mat);

        std::vector<float> norm_out(elements, 0.f);
        std::vector<float> q_out(elements, 0.f);
        std::vector<float> k_out(elements, 0.f);
        std::vector<float> v_out(elements, 0.f);
        mgr.to_row_major(fused.normalized, norm_out.data());
        mgr.to_row_major(fused.q, q_out.data());
        mgr.to_row_major(fused.k, k_out.data());
        mgr.to_row_major(fused.v, v_out.data());

        std::vector<float> ref_norm(elements, 0.f);
        std::vector<float> ref_q(elements, 0.f);
        std::vector<float> ref_k(elements, 0.f);
        std::vector<float> ref_v(elements, 0.f);
        if (rank == 0)
        {
            for (int r = 0; r < seq_len; ++r)
            {
                const float *row = act.data() + (size_t)r * hidden;
                double sum = 0.0;
                for (int c = 0; c < hidden; ++c)
                {
                    double v = row[c];
                    sum += v * v;
                }
                double inv = 1.0 / std::sqrt(sum / hidden + eps);
                for (int c = 0; c < hidden; ++c)
                    ref_norm[(size_t)r * hidden + c] = float(row[c] * inv * gamma[c]);
            }
            auto matmul_ref = [&](const std::vector<float> &a, const std::vector<float> &w, std::vector<float> &out)
            {
                for (int r = 0; r < seq_len; ++r)
                {
                    for (int c = 0; c < hidden; ++c)
                    {
                        double acc = 0.0;
                        for (int t = 0; t < hidden; ++t)
                            acc += (double)a[(size_t)r * hidden + t] * (double)w[(size_t)t * hidden + c];
                        out[(size_t)r * hidden + c] = (float)acc;
                    }
                }
            };
            matmul_ref(ref_norm, wq, ref_q);
            matmul_ref(ref_norm, wk, ref_k);
            matmul_ref(ref_norm, wv, ref_v);
        }
        MPI_Bcast(ref_norm.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(ref_q.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(ref_k.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(ref_v.data(), elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

        auto rel_l2 = [&](const std::vector<float> &a, const std::vector<float> &b)
        {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                double diff = (double)a[i] - (double)b[i];
                num += diff * diff;
                den += (double)b[i] * (double)b[i];
            }
            return std::sqrt(num / (den + 1e-30));
        };

        double rel_norm = rel_l2(norm_out, ref_norm);
        double rel_q = rel_l2(q_out, ref_q);
        double rel_k = rel_l2(k_out, ref_k);
        double rel_v = rel_l2(v_out, ref_v);
        if (rank == 0)
        {
            LOG_INFO("[ElementwiseMulti] Fused overlap=" << overlap << " rel_norm=" << rel_norm
                                                         << " rel_q=" << rel_q << " rel_k=" << rel_k << " rel_v=" << rel_v);
        }
        EXPECT_LT(rel_norm, 5e-4);
        EXPECT_LT(rel_q, 5e-4);
        EXPECT_LT(rel_k, 5e-4);
        EXPECT_LT(rel_v, 5e-4);
    }

    unsetenv("LLAMINAR_COSMA_OVERLAP_STREAM");
    done_fused.store(true);
}
