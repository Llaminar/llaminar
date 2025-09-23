// CosmaPrefillManager correctness test
#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <gtest/gtest.h>
#include <cblas.h>
#include <random>
#include <vector>
#include <cmath>

static int _early_notice = [](){ fprintf(stderr, "[TEST] static init executing before main()\n"); fflush(stderr); return 0;}();

using namespace llaminar;

struct MPIFinalizerSingle
{
    ~MPIFinalizerSingle()
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
static MPIFinalizerSingle mpi_finalizer_single;

static void fill_rand(std::vector<float> &v, float scale = 1.f)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &x : v)
        x = dist(gen);
}

TEST(CosmaPrefillManagerTest, SmallCorrectnessSingleRank)
{
    fprintf(stderr, "[TEST] Enter test body\n"); fflush(stderr);
    // Intentionally avoid MPI initialization; this test exercises single-rank fallback logic only.
    // If executed under mpirun, still allow but skip if size>1.
    int world_size_reported = 1;
    int mpi_init=0; MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_reported);
        if (world_size_reported!=1) {
            GTEST_SKIP() << "Skip multi-rank environment for single-rank test";
        }
    }

    int m = 64, k = 128, n = 96; // moderate size
    fprintf(stderr, "[TEST] dims m=%d k=%d n=%d\n", m,k,n);
    std::vector<float> A(m * k), B(k * n), C_cosma(m * n, 0.f), C_ref(m * n, 0.f);
    fill_rand(A);
    fill_rand(B);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    fprintf(stderr, "[TEST] Filled random matrices, starting reference GEMM...\n"); fflush(stderr);

    // Reference GEMM (row-major): C = A * B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A.data(), k, B.data(), n, 0.0f, C_ref.data(), n);

    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    // Avoid strategy construction for single-rank path (suspected hang point) by using generic helpers
    WeightDescriptor desc{"W", k, n, (int64_t)n, (int64_t)1, 0, B.data()};
    fprintf(stderr, "[TEST] Converting activation...\n"); fflush(stderr);
    auto A_view = mgr.convert_activation_in(A.data(), m, k);
    fprintf(stderr, "[TEST] Loading weight...\n"); fflush(stderr);
    auto W_handle = mgr.load_weight(desc);
    fprintf(stderr, "[TEST] Calling matmul...\n"); fflush(stderr);
    auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f);
    fprintf(stderr, "[TEST] Matmul returned, reconstructing...\n"); fflush(stderr);
    mgr.to_row_major(C_view, C_cosma.data());
    fprintf(stderr, "[TEST] Reconstruction done, computing rel L2...\n"); fflush(stderr);

    // Compute relative L2 error
    double num = 0.0, den = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double d = (double)C_cosma[i] - C_ref[i];
        num += d * d;
        den += (double)C_ref[i] * C_ref[i];
    }
    double rel_l2 = std::sqrt(num) / std::sqrt(den);
    fprintf(stderr, "[TEST] rel_l2=%g\n", rel_l2); fflush(stderr);
    EXPECT_LT(rel_l2, 5e-2) << "Relative L2 too large: " << rel_l2;
}
