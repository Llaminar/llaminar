#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "utils/DebugEnv.h"
#include "operators/MPIAttentionOperator.h"
#include "tensors/TensorFactory.h"
#include "Logger.h"

using namespace llaminar;

#include <mpi.h>

class MPIGuardEnvironment : public ::testing::Environment
{
public:
    void SetUp() override
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);
        }
    }
    void TearDown() override
    {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized)
            MPI_Finalize();
    }
};
::testing::Environment *const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIGuardEnvironment());

// Integration test: exercise MPIAttentionOperator::computeLocalOutputProjection with and without
// TP simulation enabled via environment flags and verify output parity.
// We restrict to single-process / single-rank usage (no MPI collectives required); the kernel's
// internal simulation path should produce identical results to baseline path.

static void fill_random(std::vector<float> &buf, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &v : buf)
        v = dist(rng);
}

static void run_case(size_t seq_len, int heads, int head_dim, int tp_parts, const char *mode_env)
{
    size_t hidden = static_cast<size_t>(heads) * head_dim;
    std::mt19937 rng(1337u + (unsigned)(seq_len + hidden));

    // Allocate pseudo local tensors (SimpleTensor via factory)
    auto local_attended = TensorFactory::create_simple({(int)seq_len, (int)hidden});
    auto local_wo = TensorFactory::create_simple({(int)hidden, (int)hidden});
    auto local_out_baseline = TensorFactory::create_simple({(int)seq_len, (int)hidden});
    auto local_out_sim = TensorFactory::create_simple({(int)seq_len, (int)hidden});

    std::vector<float> tmp((size_t)seq_len * hidden);
    std::vector<float> W((size_t)hidden * hidden);
    fill_random(tmp, rng);
    fill_random(W, rng);
    std::memcpy(local_attended->data(), tmp.data(), tmp.size() * sizeof(float));
    std::memcpy(local_wo->data(), W.data(), W.size() * sizeof(float));

    MPIAttentionOperator kernel(heads, heads, head_dim, 10000.0f, MPIAttentionOperator::DistributionStrategy::HEAD_WISE);

    // 1. Baseline (ensure tp_sim disabled)
    unsetenv("LLAMINAR_TP_WO_SIM_ENABLE");
    unsetenv("LLAMINAR_TP_WO_SIM_PARTITIONS");
    unsetenv("LLAMINAR_TP_WO_SIM_MODE");
    debugEnvRefresh();
    kernel.testInvokeOutputProjection(local_attended, local_wo, local_out_baseline, seq_len, heads, hidden);

    // 2. TP simulation enabled
    setenv("LLAMINAR_TP_WO_SIM_ENABLE", "1", 1);
    if (tp_parts > 1)
    {
        std::string p = std::to_string(tp_parts);
        setenv("LLAMINAR_TP_WO_SIM_PARTITIONS", p.c_str(), 1);
    }
    if (mode_env)
        setenv("LLAMINAR_TP_WO_SIM_MODE", mode_env, 1);
    else
        unsetenv("LLAMINAR_TP_WO_SIM_MODE");
    debugEnvRefresh();
    kernel.testInvokeOutputProjection(local_attended, local_wo, local_out_sim, seq_len, heads, hidden);

    // Compare
    const float *A = local_out_baseline->data();
    const float *B = local_out_sim->data();
    double max_abs = 0.0, num = 0.0, den = 0.0;
    for (size_t i = 0; i < tmp.size(); ++i)
    {
        double d = (double)A[i] - (double)B[i];
        if (std::fabs(d) > max_abs)
            max_abs = std::fabs(d);
        num += d * d;
        den += (double)A[i] * (double)A[i];
    }
    double rel_l2 = std::sqrt(num / (den + 1e-12));
    EXPECT_LT(max_abs, 1e-5) << "max_abs=" << max_abs << " seq_len=" << seq_len << " hidden=" << hidden << " tp_parts=" << tp_parts;
    EXPECT_LT(rel_l2, 1e-6) << "rel_l2=" << rel_l2;

    // Cleanup env for next case
    unsetenv("LLAMINAR_TP_WO_SIM_ENABLE");
    unsetenv("LLAMINAR_TP_WO_SIM_PARTITIONS");
    unsetenv("LLAMINAR_TP_WO_SIM_MODE");
    debugEnvRefresh();
}

TEST(AttentionTPSimIntegration, ColumnAndRowModes)
{
    run_case(12, 4, 32, 3, "col"); // force column
    run_case(11, 5, 24, 3, "row"); // force row
}

TEST(AttentionTPSimIntegration, AutoHeuristic)
{
    // hidden divisible by parts -> column; else seq_len divisible -> row
    run_case(17, 4, 32, 4, nullptr); // auto -> column
    run_case(18, 5, 24, 6, nullptr); // auto -> row (seq divisible by 6, hidden not)
}
