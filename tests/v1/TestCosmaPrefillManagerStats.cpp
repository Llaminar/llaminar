#include "../src/CosmaPrefillManager.h"
#include "../src/Logger.h"
#include <gtest/gtest.h>
#include <cblas.h>
#include <mpi.h>
#include <vector>
#include <cstdlib>

using namespace llaminar;

namespace
{
    struct MPIFinalizer
    {
        ~MPIFinalizer()
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
    static MPIFinalizer mpi_finalizer;

    static void fill(std::vector<float> &v)
    {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = float(i % 127) * 0.001f;
    }
}

// Test 1: Gating disabled when below threshold or explicitly disabled via env.
TEST(CosmaPrefillManagerStatsTest, GatingDisableEnv)
{
    int init = 0;
    MPI_Initialized(&init);
    if (!init)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);
    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    ASSERT_FALSE(mgr.enabled_for(100000)) << "Enabled even though ADAPTIVE_DISABLE_COSMA set";
    unsetenv("ADAPTIVE_DISABLE_COSMA");
}

// Test 2: Validation tile triggers counter increment (multi-rank only). If single rank, skip.
TEST(CosmaPrefillManagerStatsTest, ValidationTileIncrements)
{
    int init = 0;
    MPI_Initialized(&init);
    if (!init)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world < 2)
        GTEST_SKIP() << "Requires >=2 ranks to exercise validation tile";

    setenv("LLAMINAR_COSMA_PREFILL_THRESHOLD", "1", 1); // ensure enabled_for
    setenv("LLAMINAR_COSMA_VALIDATE_TILE", "8", 1);     // tile size

    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();

    int m = 16, k = 32, n = 24; // small but > tile
    std::vector<float> A(m * k, 0.f), B(k * n, 0.f), C(m * n, 0.f);
    fill(A);
    fill(B);

    // Force COSMA path through AdaptiveMatMulManager selection logic would be complex; invoke directly.
    WeightDescriptor desc{"Wstats", k, n, (int64_t)n, 1, 0, B.data()};
    auto A_view = mgr.convert_activation_in(A.data(), m, k);
    auto W_handle = mgr.load_weight(desc);
    // Matmul (should engage COSMA if world>1 and threshold met). Even if small volume triggers fast path,
    // validation only runs when world>1, so we bump volume by replicating logic: call matmul then rely on path.
    auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(C_view, C.data());

    // After matmul, validation counter should be >=1
    const auto &stats = mgr.stats();
    EXPECT_GE(stats.validation_tile_checks.load(), 1) << "Validation tile did not run";

    unsetenv("LLAMINAR_COSMA_VALIDATE_TILE");
}

// Test 3: Bytes counters increase after streaming + activation conversion (multi-rank) or are zero single rank.
TEST(CosmaPrefillManagerStatsTest, BytesCounters)
{
    int init = 0;
    MPI_Initialized(&init);
    if (!init)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Provide threshold low so enabled_for would be true if multi-rank
    setenv("LLAMINAR_COSMA_PREFILL_THRESHOLD", "1", 1);

    CosmaPrefillManager &mgr = CosmaPrefillManager::instance();
    long long before_act = mgr.stats().bytes_converted_activations.load();
    long long before_w = mgr.stats().bytes_streamed_weights.load();

    int m = 32, k = 16, n = 20;
    std::vector<float> A(m * k, 0.5f), B(k * n, 0.25f), C(m * n, 0.f);
    WeightDescriptor desc{"Wbytes", k, n, (int64_t)n, 1, 0, B.data()};
    auto A_view = mgr.convert_activation_in(A.data(), m, k);
    auto W_handle = mgr.load_weight(desc);
    auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(C_view, C.data());

    long long after_act = mgr.stats().bytes_converted_activations.load();
    long long after_w = mgr.stats().bytes_streamed_weights.load();

    if (world > 1)
    {
        EXPECT_GT(after_act, before_act) << "Activation bytes not recorded";
        EXPECT_GT(after_w, before_w) << "Weight bytes not recorded";
    }
    else
    {
        // single rank path doesn't allocate COSMA matrices -> counters may remain unchanged
        SUCCEED();
    }
}
