#include "../src/CosmaPrefillManager.h"
#include "../src/Logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <cblas.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <iostream>

using namespace llaminar;

namespace
{
    struct MPIFinalizerPerf
    {
        ~MPIFinalizerPerf()
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
    static MPIFinalizerPerf _mpi_finalizer_perf;

    void fill_rand(std::vector<float> &v, int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto &x : v)
            x = dist(gen);
    }

    struct Shape
    {
        int m, k, n;
    };

    struct Metrics
    {
        double us_popA = 0, us_popB = 0, us_matmul = 0, us_total = 0;
        double rel_l2 = 0;
        double gflops = 0;
        size_t bytesA = 0, bytesB = 0;
        std::string variant;
        Shape s;
    };

    inline double us_since(const std::chrono::high_resolution_clock::time_point &t0)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count();
    }

    void run_variant(std::vector<Metrics> &out, const Shape &shape, bool legacy_forward, int world, int rank)
    {
        int m = shape.m, k = shape.k, n = shape.n;
        // Prepare inputs replicated
        std::vector<float> A(m * k), B(k * n), C(m * n, 0.f), C_ref(m * n, 0.f);
        if (rank == 0)
        {
            fill_rand(A, 100 + m + k + n);
            fill_rand(B, 200 + m + k + n);
        }
        MPI_Bcast(A.data(), m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(B.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Reference GEMM on rank 0 only
        if (rank == 0)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.f, A.data(), k, B.data(), n, 0.f, C_ref.data(), n);
        }
        MPI_Bcast(C_ref.data(), m * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (legacy_forward)
            setenv("LLAMINAR_COSMA_POP_FORWARD_LEGACY", "1", 1);
        else
            unsetenv("LLAMINAR_COSMA_POP_FORWARD_LEGACY");

        auto &mgr = CosmaPrefillManager::instance();
        mgr.reset_stats();
        auto &strat = mgr.strategy_for(m, n, k);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_total0 = std::chrono::high_resolution_clock::now();
        auto t0 = std::chrono::high_resolution_clock::now();
        auto A_view = mgr.convert_activation_in_with_strategy(A.data(), m, k, strat);
        auto t1 = std::chrono::high_resolution_clock::now();
        WeightDescriptor desc{"W_perf", k, n, (int64_t)n, (int64_t)1, 0, B.data()};
        auto W_handle = mgr.load_weight_with_strategy(desc, strat);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f);
        auto t3 = std::chrono::high_resolution_clock::now();
        mgr.to_row_major(C_view, C.data());
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_total1 = std::chrono::high_resolution_clock::now();

        // rel L2
        double num = 0, den = 0;
        for (int i = 0; i < m * n; ++i)
        {
            double d = (double)C[i] - C_ref[i];
            num += d * d;
            den += (double)C_ref[i] * C_ref[i];
        }
        double rel_l2 = (den > 0) ? std::sqrt(num) / std::sqrt(den + 1e-30) : 0.0;
        double flops = 2.0 * (double)m * n * k;   // multiply-add
        double ms_matmul = us_since(t2) / 1000.0; // (since t2 to t3 microseconds) but we captured t2 and t3
        double us_popA = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        double us_popB = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        double us_matmul = (double)std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        double us_total = (double)std::chrono::duration_cast<std::chrono::microseconds>(t_total1 - t_total0).count();
        double gflops = (us_matmul > 0) ? flops / (us_matmul * 1e6) : 0.0;

        Metrics mrec;
        mrec.us_popA = us_popA;
        mrec.us_popB = us_popB;
        mrec.us_matmul = us_matmul;
        mrec.us_total = us_total;
        mrec.rel_l2 = rel_l2;
        mrec.gflops = gflops;
        mrec.bytesA = (size_t)m * k * sizeof(float);
        mrec.bytesB = (size_t)k * n * sizeof(float);
        mrec.variant = legacy_forward ? "legacy_forward" : "dest_local";
        mrec.s = shape;
        out.push_back(mrec);
    }

}

TEST(CosmaPrefillManagerPopulationPerfTest, PopulationBenchmark)
{
    if (!std::getenv("LLAMINAR_COSMA_POP_BENCH"))
    {
        GTEST_SKIP() << "Set LLAMINAR_COSMA_POP_BENCH=1 to run population performance benchmark";
    }
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
        GTEST_SKIP() << "Requires >=2 ranks";
    }

    std::vector<Shape> shapes = {{192, 192, 256}, {256, 256, 256}, {384, 384, 384}};
    std::vector<Metrics> metrics;
    metrics.reserve(shapes.size() * 2);

    for (auto &s : shapes)
    {
        run_variant(metrics, s, false, world, rank); // dest-local
        run_variant(metrics, s, true, world, rank);  // legacy
    }

    if (rank == 0)
    {
        std::cout << "#COSMA_POPULATION_BENCHMARK CSV\n";
        std::cout << "m,k,n,variant,popA_us,popB_us,matmul_us,total_us,rel_l2,GFLOPS,bytesA,bytesB\n";
        for (auto &mrec : metrics)
        {
            std::cout << mrec.s.m << ',' << mrec.s.k << ',' << mrec.s.n << ',' << mrec.variant << ','
                      << (long long)mrec.us_popA << ',' << (long long)mrec.us_popB << ',' << (long long)mrec.us_matmul << ','
                      << (long long)mrec.us_total << ',' << mrec.rel_l2 << ',' << mrec.gflops << ','
                      << mrec.bytesA << ',' << mrec.bytesB << "\n";
        }
        // Basic sanity: rel_l2 should be modest (<2e-2) for both variants
        for (auto &mrec : metrics)
        {
            EXPECT_LT(mrec.rel_l2, 2e-2) << "rel_l2 too large for variant " << mrec.variant;
        }
    }
}
