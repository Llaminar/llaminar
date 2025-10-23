#include "../src/CosmaPrefillManager.h"
#include "../src/Logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <cblas.h>
#include <vector>
#include <random>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>
#include <execinfo.h>
#include <csignal>
#include <unistd.h>

using namespace llaminar;

struct MPIFinalizerMulti
{
    ~MPIFinalizerMulti()
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
static MPIFinalizerMulti mpi_finalizer_multi;

static void fill_rand(std::vector<float> &v, float scale = 1.f)
{
    std::mt19937 gen(12345 + (int)v.size());
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &x : v)
        x = dist(gen);
}

TEST(CosmaPrefillManagerMultiTest, SmallFastPathDistributed)
{
    // ---------------- Watchdog with rank-local stack dump ----------------
    static std::atomic<bool> done{false};
    int internal_timeout_ms = 30000; // must be < CTest TIMEOUT (60s)
    if (const char *env = std::getenv("LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS"))
    {
        int v = std::atoi(env);
        if (v > 0)
            internal_timeout_ms = v;
        else if (v == 0)
            internal_timeout_ms = -1; // 0 disables
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
    std::thread watchdog([&]()
                         {
        if (internal_timeout_ms < 0) return; // disabled
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_tp).count();
            if (elapsed > internal_timeout_ms && !done.load()) {
                int r=0, w=1; if (MPI_Initialized(nullptr)) { MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&w);} 
                fprintf(stderr, "[WATCHDOG][rank %d/%d] Elapsed %lld ms > %d ms. Dumping stack and aborting.\n", r,w,(long long)elapsed, internal_timeout_ms); fflush(stderr);
                stack_dump(r);
                // Use abort so CTest registers failure quickly with stack output preserved
                ::raise(SIGABRT);
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
    // ---------------------------------------------------------------------
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

    // Choose dimensions below fast path threshold (default 64^3=262144). Use something small.
    int m = 32, k = 48, n = 40; // volume = 61,440 < 262,144
    std::vector<float> A(m * k), B(k * n), C_result(m * n, 0.f), C_ref(m * n, 0.f);
    fill_rand(A);
    fill_rand(B);

    // Each rank uses identical A,B (Phase 1 replication assumption)
    // Reference GEMM computed only on rank 0
    if (rank == 0)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.f, A.data(), k, B.data(), n, 0.f, C_ref.data(), n);
    }
    // Broadcast reference so all ranks can compare after collect (optional)
    MPI_Bcast(C_ref.data(), m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Force fast path use by ensuring threshold stays default (or higher) - no env override inside test
    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    auto &strat = mgr.strategy_for(m, n, k); // ensures strategy cached (even if not used)
    (void)strat;

    WeightDescriptor desc{"W", k, n, (int64_t)n, 1, 0, B.data()};
    auto A_view = mgr.convert_activation_in_with_strategy(A.data(), m, k, mgr.strategy_for(m, k, k));
    auto W_handle = mgr.load_weight_with_strategy(desc, mgr.strategy_for(k, n, k));
    auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f); // Potential hang site
    mgr.to_row_major(C_view, C_result.data());

    // Compute relative L2 (should match within small epsilon ~1e-5 since identical ops)
    double num = 0.0, den = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double d = (double)C_result[i] - C_ref[i];
        num += d * d;
        den += (double)C_ref[i] * C_ref[i];
    }
    double rel_l2 = std::sqrt(num) / std::sqrt(den + 1e-30);

    if (rank == 0)
    {
        LOG_INFO("[Test] Multi-rank fast path rel_l2=" << rel_l2);
    }
    EXPECT_LT(rel_l2, 1e-4) << "Rel L2 too large for fast path distributed fallback: " << rel_l2;
    done.store(true); // signal watchdog to exit normally
}
