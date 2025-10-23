#include "../src/CosmaPrefillManager.h"
#include "../src/Logger.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
using namespace llaminar;

// This test simulates a potential orientation mismatch and ensures that enabling
// LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE does not degrade correctness vs explicit transpose.
TEST(CosmaPrefillManagerOrientationAutoFixTest, AutoFixDoesNotWorsenResult)
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
    int m = 16, k = 16, n = 16; // square for simplicity
    std::vector<float> A(m * k), B(k * n), C1(m * n), C2(m * n);
    for (int i = 0; i < m * k; ++i)
        A[i] = std::sin(i * 0.01f);
    for (int i = 0; i < k * n; ++i)
        B[i] = std::cos(i * 0.02f);
    WeightDescriptor desc{"Worient", k, n, (int64_t)n, 1, 0, B.data()};
    // First run baseline without auto-fix
    unsetenv("LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE");
    auto A_view1 = mgr.convert_activation_in(A.data(), m, k);
    auto W1 = mgr.load_weight(desc);
    auto C_view1 = mgr.matmul(A_view1, W1, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(C_view1, C1.data());
    // Second run with auto-fix enabled
    setenv("LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE", "1", 1);
    auto A_view2 = mgr.convert_activation_in(A.data(), m, k);
    auto W2 = mgr.load_weight(desc);
    auto C_view2 = mgr.matmul(A_view2, W2, m, k, n, false, 1.f, 0.f);
    mgr.to_row_major(C_view2, C2.data());
    // Compare relative L2 difference between outputs (should be small)
    double num = 0.0, den = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double d = (double)C1[i] - (double)C2[i];
        num += d * d;
        den += (double)C1[i] * C1[i];
    }
    double rel_l2 = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    EXPECT_LT(rel_l2, 1e-4) << "Auto-fix altered results unexpectedly rel_l2=" << rel_l2;
}
