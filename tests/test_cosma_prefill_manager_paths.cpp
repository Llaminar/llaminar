#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <cblas.h>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace llaminar;

namespace {
struct MPIFinalizerPaths {
    ~MPIFinalizerPaths(){ int init=0,fin=0; MPI_Initialized(&init); if(init){ MPI_Finalized(&fin); if(!fin) MPI_Finalize(); }}
};
static MPIFinalizerPaths _mpi_finalizer_paths;

static void fill(std::vector<float>& v){ for(size_t i=0;i<v.size();++i) v[i] = float((i*1315423911u) % 1000) * 0.0005f; }

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b){ double num=0,den=0; for(size_t i=0;i<a.size();++i){ double d = double(a[i])-double(b[i]); num += d*d; den += double(b[i])*double(b[i]); } return std::sqrt(num)/(std::sqrt(den)+1e-30); }
}

// Helper: ensure MPI initialized for multi-rank tests
static void ensure_mpi(){ int init=0; MPI_Initialized(&init); if(!init){ int argc=0; char** argv=nullptr; MPI_Init(&argc,&argv);} }

TEST(CosmaPrefillManagerPathsTest, DirectCompare)
{
    ensure_mpi();
    int world=1; MPI_Comm_size(MPI_COMM_WORLD,&world); if(world<2) GTEST_SKIP() << "Need >=2 ranks";
    setenv("LLAMINAR_COSMA_FORCE_DIRECT","1",1);
    setenv("LLAMINAR_COSMA_COMPARE_REPLICATED","1",1);
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG");

    int m=96,k=96,n=128; // volume above fast path? 96*96*128=1,179,648 (>64^3=262,144) ensures not fast path
    std::vector<float> A(m*k,0.f), B(k*n,0.f), C(m*n,0.f), Cref(m*n,0.f);
    fill(A); fill(B);
    // Broadcast identical inputs so repl compare uses same data
    MPI_Bcast(A.data(), m*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(MPI::COMM_WORLD.Get_rank()==0){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.f,A.data(),k,B.data(),n,0.f,Cref.data(),n);
    }
    MPI_Bcast(Cref.data(), m*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto &mgr = CosmaPrefillManager::instance();
    long long before_cosma = mgr.stats().cosma_path_calls.load();
    const auto &strat = mgr.strategy_for(m,n,k);
    auto Av = mgr.convert_activation_in_with_strategy(A.data(), m, k, strat);
    WeightDescriptor desc{"Wdc", k, n, (int64_t)n, 1, 0, B.data()};
    auto W = mgr.load_weight_with_strategy(desc, strat);
    auto Cv = mgr.matmul(Av, W, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(Cv, C.data());
    double r = rel_l2(C, Cref);
    EXPECT_LT(r, 2e-2) << "Rel L2 too large for direct compare path";
    EXPECT_GT(mgr.stats().cosma_path_calls.load(), before_cosma);

    unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
    unsetenv("LLAMINAR_COSMA_COMPARE_REPLICATED");
}

TEST(CosmaPrefillManagerPathsTest, ForceReplicatedDiag)
{
    ensure_mpi();
    int world=1; MPI_Comm_size(MPI_COMM_WORLD,&world); if(world<2) GTEST_SKIP() << "Need >=2 ranks";
    setenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG","1",1);
    unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
    unsetenv("LLAMINAR_COSMA_COMPARE_REPLICATED");

    int m=32,k=40,n=48; // below fast path threshold anyway but forced diag path should still count as fast path
    std::vector<float> A(m*k,0.f), B(k*n,0.f), C(m*n,0.f), Cref(m*n,0.f);
    fill(A); fill(B);
    MPI_Bcast(A.data(), m*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(MPI::COMM_WORLD.Get_rank()==0){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.f,A.data(),k,B.data(),n,0.f,Cref.data(),n);
    }
    MPI_Bcast(Cref.data(), m*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto &mgr = CosmaPrefillManager::instance();
    long long before_fast = mgr.stats().fast_path_calls.load();
    auto Av = mgr.convert_activation_in(A.data(), m, k);
    WeightDescriptor desc{"Wrdiag", k, n, (int64_t)n, 1, 0, B.data()};
    auto W = mgr.load_weight(desc);
    auto Cv = mgr.matmul(Av, W, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(Cv, C.data());
    double r = rel_l2(C, Cref);
    EXPECT_LT(r, 2e-2);
    EXPECT_GT(mgr.stats().fast_path_calls.load(), before_fast);

    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG");
}

TEST(CosmaPrefillManagerPathsTest, FastPathUnverified)
{
    ensure_mpi();
    int world=1; MPI_Comm_size(MPI_COMM_WORLD,&world); if(world<2) GTEST_SKIP() << "Need >=2 ranks";
    setenv("LLAMINAR_COSMA_VALIDATE_TILE","8",1);
    setenv("LLAMINAR_COSMA_FAST_UNVERIFIED","1",1);
    int m=16,k=24,n=20; // very small
    std::vector<float> A(m*k,0.f), B(k*n,0.f), C(m*n,0.f), Cref(m*n,0.f);
    fill(A); fill(B);
    MPI_Bcast(A.data(), m*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(MPI::COMM_WORLD.Get_rank()==0){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.f,A.data(),k,B.data(),n,0.f,Cref.data(),n);
    }
    MPI_Bcast(Cref.data(), m*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto &mgr = CosmaPrefillManager::instance();
    long long before_val = mgr.stats().validation_tile_checks.load();
    auto Av = mgr.convert_activation_in(A.data(), m, k);
    WeightDescriptor desc{"Wfp", k, n, (int64_t)n, 1, 0, B.data()};
    auto W = mgr.load_weight(desc);
    auto Cv = mgr.matmul(Av, W, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(Cv, C.data());
    double r = rel_l2(C, Cref);
    EXPECT_LT(r, 5e-2);
    // validation tile should not have run due to FAST_UNVERIFIED
    EXPECT_EQ(mgr.stats().validation_tile_checks.load(), before_val);

    unsetenv("LLAMINAR_COSMA_VALIDATE_TILE");
    unsetenv("LLAMINAR_COSMA_FAST_UNVERIFIED");
}

TEST(CosmaPrefillManagerPathsTest, AutoDirectThreshold)
{
    ensure_mpi();
    int world=1; MPI_Comm_size(MPI_COMM_WORLD,&world); if(world<2) GTEST_SKIP() << "Need >=2 ranks";
    unsetenv("LLAMINAR_COSMA_FORCE_DIRECT");
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED");
    unsetenv("LLAMINAR_COSMA_FORCE_REPLICATED_DIAG");
    unsetenv("LLAMINAR_COSMA_COMPARE_REPLICATED");

    // Choose a shape likely above auto direct threshold (if threshold large, test still valid with force via volume heuristic).
    int m=160,k=192,n=144; // volume ~4.4M
    std::vector<float> A(m*k,0.f), B(k*n,0.f), C(m*n,0.f), Cref(m*n,0.f);
    fill(A); fill(B);
    MPI_Bcast(A.data(), m*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), k*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(MPI::COMM_WORLD.Get_rank()==0){
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.f,A.data(),k,B.data(),n,0.f,Cref.data(),n);
    }
    MPI_Bcast(Cref.data(), m*n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto &mgr = CosmaPrefillManager::instance();
    long long before_cosma = mgr.stats().cosma_path_calls.load();
    auto Av = mgr.convert_activation_in(A.data(), m, k);
    WeightDescriptor desc{"Wauto", k, n, (int64_t)n, 1, 0, B.data()};
    auto W = mgr.load_weight(desc);
    auto Cv = mgr.matmul(Av, W, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(Cv, C.data());
    double r = rel_l2(C, Cref);
    EXPECT_LT(r, 2e-2);
    EXPECT_GT(mgr.stats().cosma_path_calls.load(), before_cosma) << "Expected COSMA direct path based on volume";
}
