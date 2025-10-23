#include <gtest/gtest.h>
#include "TestTimeoutGuard.h"
#include "../src/AdaptiveMatmul.h"
#include <mpi.h>
#include <chrono>
#include <random>

using namespace llaminar;

namespace
{
    std::vector<float> make_matrix(int rows, int cols, float lo = -0.05f, float hi = 0.05f)
    {
        std::vector<float> v(rows * cols);
        std::mt19937 gen(42 + rows + cols);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto &x : v)
            x = dist(gen);
        return v;
    }
}

class AdaptiveMatMulBatchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int init;
        MPI_Initialized(&init);
        if (!init)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);
        }
        mgr_ = std::make_unique<AdaptiveMatMulManager>();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }
    void TearDown() override { mgr_.reset(); }
    std::unique_ptr<AdaptiveMatMulManager> mgr_;
    int rank_, size_;
};

TEST_F(AdaptiveMatMulBatchTest, BatchStaysOpenBLASBelowThreshold)
{
    // Three small GEMMs (m < 4096) should NOT invoke COSMA.
    const int k = 128, n = 256;
    int ms[3] = {64, 128, 96};
    std::vector<std::vector<float>> A_store;
    A_store.reserve(3);
    std::vector<std::vector<float>> C_store(3);
    for (int i = 0; i < 3; ++i)
    {
        A_store.push_back(make_matrix(ms[i], k));
        C_store[i].assign(ms[i] * n, 0.0f);
    }
    auto B = make_matrix(k, n);
    std::vector<const float *> A_ptrs(3);
    std::vector<float *> C_ptrs(3);
    for (int i = 0; i < 3; ++i)
    {
        A_ptrs[i] = A_store[i].data();
        C_ptrs[i] = C_store[i].data();
    }
    ASSERT_TRUE(mgr_->multiply_batch(A_ptrs.data(), B.data(), C_ptrs.data(), ms, 3, n, k, true));
    // last_backend should be OPENBLAS
    EXPECT_EQ(mgr_->last_backend(), MatMulBackend::OPENBLAS);
}

TEST_F(AdaptiveMatMulBatchTest, SingleLargeDefersToNormalPath)
{
    // Single large prefill candidate (m=4096) should route through standard path and potentially COSMA if multi-rank.
    const int m = 4096, k = 64, n = 64;
    auto A = make_matrix(m, k);
    auto B = make_matrix(k, n);
    std::vector<float> C(m * n, 0.0f);
    const float *A_list[1] = {A.data()};
    float *C_list[1] = {C.data()};
    int ms[1] = {m};
    ASSERT_TRUE(mgr_->multiply_batch(A_list, B.data(), C_list, ms, 1, n, k, true));
    if (size_ >= 2)
    {
        EXPECT_EQ(mgr_->last_backend(), MatMulBackend::COSMA);
    }
    else
    {
        EXPECT_EQ(mgr_->last_backend(), MatMulBackend::OPENBLAS);
    }
}

// Explicit gating test: ensure COSMA never used when m < 4096 even if prefill flag is set.
TEST_F(AdaptiveMatMulBatchTest, ForceOpenBLASBelowThreshold)
{
    const int m = 2048, k = 256, n = 256;
    auto A = make_matrix(m, k);
    auto B = make_matrix(k, n);
    std::vector<float> C(m * n, 0.0f);
    ASSERT_TRUE(mgr_->multiply(A.data(), B.data(), C.data(), m, n, k, false, false, 1.0f, 0.0f, true));
    EXPECT_EQ(mgr_->last_backend(), MatMulBackend::OPENBLAS);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    auto timeout = llaminar::test_util::TestTimeoutGuard::ResolveTimeout(
        {"LLAMINAR_TEST_TIMEOUT_MS"}, std::chrono::milliseconds(60000));
    llaminar::test_util::TestTimeoutGuard watchdog("AdaptiveMatMulBatchTest", timeout);
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();
    watchdog.disarm();
    MPI_Finalize();
    return rc;
}
