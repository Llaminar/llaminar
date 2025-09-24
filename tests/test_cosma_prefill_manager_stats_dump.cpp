#include "../src/cosma_prefill_manager.h"
#include "../src/logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <fstream>
#include <regex>
#include <cstdio>
#include <cstdlib>
using namespace llaminar;

TEST(CosmaPrefillManagerStatsDumpTest, JsonFileContainsCoreFields)
{
    int init = 0;
    MPI_Initialized(&init);
    if (!init)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    auto &mgr = CosmaPrefillManager::instance();
    mgr.reset_stats();
    // Minimal operation to populate a few counters (single rank safe)
    int m = 4, k = 4, n = 4;
    std::vector<float> A(m * k, 1.f), B(k * n, 2.f), C(m * n, 0.f);
    WeightDescriptor desc{"Wdump", k, n, (int64_t)n, 1, 0, B.data()};
    auto A_view = mgr.convert_activation_in(A.data(), m, k);
    auto W_handle = mgr.load_weight(desc);
    auto C_view = mgr.matmul(A_view, W_handle, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(C_view, C.data());
    std::string path = "cosma_prefill_stats_test.json";
    mgr.dump_stats_json(path);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::ifstream ifs(path);
        ASSERT_TRUE(ifs.is_open());
        std::string content((std::istreambuf_iterator<char>(ifs)), {});
        // Basic key presence checks
        EXPECT_NE(content.find("\"fast_path_calls\""), std::string::npos);
        EXPECT_NE(content.find("\"cosma_path_calls\""), std::string::npos);
        EXPECT_NE(content.find("\"peak_resident_bytes\""), std::string::npos);
    }
}

TEST(CosmaPrefillManagerStatsDumpTest, EnvFlagHonoursCustomPath)
{
    int init = 0;
    MPI_Initialized(&init);
    if (!init)
    {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    auto &mgr = CosmaPrefillManager::instance();
    mgr.reset_stats();
    const char *path = "cosma_prefill_stats_env.json";
    std::remove(path);
    setenv("LLAMINAR_COSMA_DUMP_STATS", "1", 1);
    setenv("LLAMINAR_COSMA_DUMP_STATS_PATH", path, 1);
    mgr.dump_stats_if_requested();
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::ifstream ifs(path);
        ASSERT_TRUE(ifs.is_open());
    }
    unsetenv("LLAMINAR_COSMA_DUMP_STATS");
    unsetenv("LLAMINAR_COSMA_DUMP_STATS_PATH");
}
