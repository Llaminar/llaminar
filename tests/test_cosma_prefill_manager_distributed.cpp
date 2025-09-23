// COSMA Prefill Manager Distributed GEMM Correctness Test (full COSMA path)
// This test exercises the multi-rank cosma::multiply path (NOT the fast path
// local BLAS fallback) by selecting matrix dimensions whose volume exceeds the
// fast_path_threshold_ops_ default (64^3).
// Strategy consistency is critical: we intentionally construct A, B using the
// SAME strategy (m,n,k) to avoid mismatched block-cyclic layouts that would
// otherwise produce incorrect results.

#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
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

namespace {
struct MPIFinalizerFull
{
    ~MPIFinalizerFull()
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
static MPIFinalizerFull mpi_finalizer_full;

void fill_rand(std::vector<float> &v, int seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : v)
        x = dist(gen);
}
} // namespace

TEST(CosmaPrefillManagerDistributedFullTest, LargeMatMulDistributedCorrectness)
{
    // ---------------- Watchdog with rank-local stack dump ----------------
    static std::atomic<bool> done{false};
    int internal_timeout_ms = 45000; // < external 60s CTest timeout
    if (const char* env = std::getenv("LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS")) {
        int v = std::atoi(env); if (v > 0) internal_timeout_ms = v; else if (v==0) internal_timeout_ms = -1; // 0 disables
    }
    auto start_tp = std::chrono::steady_clock::now();
    auto stack_dump = [](int rank){
        void* frames[64]; int n = ::backtrace(frames, 64); char** syms = ::backtrace_symbols(frames, n);
        fprintf(stderr, "[WATCHDOG-FULL][rank %d] === STACK TRACE (%d frames) ===\n", rank, n);
        for(int i=0;i<n;i++) fprintf(stderr, "[WATCHDOG-FULL][rank %d] %s\n", rank, syms[i]);
        if (syms) free(syms); fflush(stderr);
    };
    std::thread watchdog([&](){
        if (internal_timeout_ms < 0) return;
        while(!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_tp).count();
            if (elapsed > internal_timeout_ms && !done.load()) {
                int r=0,w=1; if (MPI_Initialized(nullptr)) { MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&w);} 
                fprintf(stderr, "[WATCHDOG-FULL][rank %d/%d] Elapsed %lld ms > %d ms. Dumping stack and aborting.\n", r,w,(long long)elapsed, internal_timeout_ms); fflush(stderr);
                stack_dump(r);
                ::raise(SIGABRT);
            }
        }
    });
    struct WatchdogJoin { std::thread &t; ~WatchdogJoin(){ done.store(true); if (t.joinable()) t.join(); } } _wd_join{watchdog};
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
    {
        GTEST_SKIP() << "Need >=2 ranks for distributed COSMA path";
    }

    // Choose dimensions giving volume >> fast path threshold (64^3=262,144)
    // Use moderate size to keep test runtime manageable.
    const int m = 256;
    const int k = 256;
    const int n = 256; // volume = 16,777,216

    // Ensure we do not accidentally force replicated path
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
    unsetenv("LLAMINAR_COSMA_FAST_UNVERIFIED");

    std::vector<float> A(m * k), B(k * n), C_result(m * n, 0.f), C_ref(m * n, 0.f);
    fill_rand(A, 123 + rank); // each rank different seeds is fine due to replicated assumption? For distributed correctness we want identical; broadcast.
    fill_rand(B, 456 + rank);
    // Replicate inputs across ranks (Phase 1 assumption): broadcast rank 0's data.
    MPI_Bcast(A.data(), m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Reference on rank 0 only
    if (rank == 0)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.f, A.data(), k, B.data(), n, 0.f, C_ref.data(), n);
    }
    MPI_Bcast(C_ref.data(), m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto &mgr = CosmaPrefillManager::instance();
    // Unified strategy for (m,n,k)
    const auto &strat = mgr.strategy_for(m, n, k);
    // Activation & weight must share SAME strategy for correctness.
    auto A_view = mgr.convert_activation_in_with_strategy(A.data(), m, k, strat);
    WeightDescriptor desc{"W_large", k, n, (int64_t)n, (int64_t)1, 0, B.data()};
    auto W_handle = mgr.load_weight_with_strategy(desc, strat);

    auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(C_view, C_result.data());

    // Relative L2 norm
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
        LOG_INFO("[Test] Distributed COSMA path rel_l2=" << rel_l2);
    }
    // Allow slightly looser tolerance vs single-rank due to potential summation order differences.
    EXPECT_LT(rel_l2, 2e-2) << "Relative L2 too large for distributed COSMA path: " << rel_l2;
    done.store(true);
}
