#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <cblas.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>

using namespace llaminar;

namespace {
void fill_rand(std::vector<float> &v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : v) x = dist(gen);
}
}

// Regression: destination-local population must reproduce original row-major
// matrices exactly (bitwise) for deterministic inputs under unified strategy.
// We verify by allocating A and B with a chosen strategy, then reconstructing
// them back to row-major via reconstruct_matrix() and comparing.
namespace { struct MPIFinalizerPop { ~MPIFinalizerPop(){ int init=0,fin=0; MPI_Initialized(&init); if(init){ MPI_Finalized(&fin); if(!fin) MPI_Finalize(); } } }; static MPIFinalizerPop _mpi_finalizer_pop; }

TEST(CosmaPrefillManagerPopulationRegressionTest, DestinationLocalExactRoundTrip) {
    int initialized = 0; MPI_Initialized(&initialized); if (!initialized) { int argc=0; char** argv=nullptr; MPI_Init(&argc,&argv); }
    int world=1, rank=0; MPI_Comm_size(MPI_COMM_WORLD,&world); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (world < 2) {
        GTEST_SKIP() << "Need >=2 ranks to exercise distributed ownership mapping";
    }

    // Purge legacy override if user set it; we want default path.
    unsetenv("LLAMINAR_COSMA_POP_FORWARD_LEGACY");
    unsetenv("LLAMINAR_COSMA_POP_DEST_LOCAL"); // historical no-op

    // Choose a shape large enough to trigger distributed path but smaller than previous full test
    // Choose dimensions aligned with strategy bucketing to avoid internal padding effects.
    const int m = 192;
    const int k = 192; // aligned (no padding)
    const int n = 256; // aligned (no padding)

    std::vector<float> A(m*k), B(k*n);
    if (rank==0) {
        fill_rand(A, 111);
        fill_rand(B, 222);
    }
    MPI_Bcast(A.data(), m*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto &mgr = CosmaPrefillManager::instance();
    const auto &strat = mgr.strategy_for(m, n, k);

    auto A_view = mgr.convert_activation_in_with_strategy(A.data(), m, k, strat);
    WeightDescriptor desc{"W_regress", k, n, (int64_t)n, (int64_t)1, 0, B.data()};
    auto W_handle = mgr.load_weight_with_strategy(desc, strat);

    // Reconstruct A & B from distributed layout back to row-major
    std::vector<float> A_round(m*k, 0.f), B_round(k*n, 0.f);
    mgr.to_row_major(A_view, A_round.data());
    mgr.to_row_major(W_handle.view, B_round.data());

    // Compute rel L2 and also count mismatches (bitwise)
    size_t mismatchesA = 0, mismatchesB = 0; double numA=0, denA=0, numB=0, denB=0;
    for (int i=0;i<m*k;++i){ double d=(double)A_round[i]-A[i]; numA+=d*d; denA+=(double)A[i]*A[i]; if (A_round[i]!=A[i]) ++mismatchesA; }
    for (int i=0;i<k*n;++i){ double d=(double)B_round[i]-B[i]; numB+=d*d; denB+=(double)B[i]*B[i]; if (B_round[i]!=B[i]) ++mismatchesB; }
    double relA = (denA>0)? std::sqrt(numA)/std::sqrt(denA+1e-30) : 0.0;
    double relB = (denB>0)? std::sqrt(numB)/std::sqrt(denB+1e-30) : 0.0;

    if (rank==0) {
        LOG_INFO("[PopulationRegression] relA="<<relA<<" mismatchesA="<<mismatchesA
                 <<" relB="<<relB<<" mismatchesB="<<mismatchesB);
    }

    // Expect exact reconstruction; allow a tiny epsilon for floating compare paranoia.
    EXPECT_LT(relA, 1e-7) << "Activation population round-trip mismatch relA="<<relA;
    EXPECT_LT(relB, 1e-7) << "Weight population round-trip mismatch relB="<<relB;
    EXPECT_EQ(mismatchesA, (size_t)0) << "Activation round-trip had bitwise mismatches";
    EXPECT_EQ(mismatchesB, (size_t)0) << "Weight round-trip had bitwise mismatches";
}
