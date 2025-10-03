/**
 * @file test_rmsnorm_core_parity.cpp
 * @brief Regression test ensuring RMSNorm core produces identical results across execution toggles
 *        (TLS scratch reuse, forced scalar path, disabled scratch) to guard against future refactors.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>

#include "kernels/common/rmsnorm_core.h"
#include "utils/debug_env.h"

using namespace llaminar::kernels;

namespace
{

    // Compute relative L2 difference between two vectors.
    double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
        {
            // Return large value; individual test will flag mismatch contextually.
            return 1e9;
        }
        long double num = 0.0L;
        long double denom = 0.0L;
        for (size_t i = 0; i < a.size(); ++i)
        {
            long double da = a[i];
            long double db = b[i];
            long double diff = da - db;
            num += diff * diff;
            denom += da * da;
        }
        if (denom < 1e-30L)
            denom = 1.0L;
        return std::sqrt((double)num) / std::sqrt((double)denom);
    }

    struct RunConfig
    {
        bool force_scalar = false;
        bool disable_tls_scratch = false;
        int prealloc_rows = 0;
    };

    // Helper to clear RMSNorm-related env vars so each run is isolated.
    void clear_rmsnorm_env()
    {
        unsetenv("LLAMINAR_RMSNORM_FORCE_SCALAR");
        unsetenv("LLAMINAR_RMSNORM_DISABLE_TLS_SCRATCH");
        unsetenv("LLAMINAR_RMSNORM_SCRATCH_PREALLOC_ROWS");
        unsetenv("LLAMINAR_RMSNORM_FALSE_SHARING_PROBE");
    }

    std::vector<float> run_case(size_t rows, size_t cols, const RunConfig &cfg, unsigned seed)
    {
        clear_rmsnorm_env();
        if (cfg.force_scalar)
            setenv("LLAMINAR_RMSNORM_FORCE_SCALAR", "1", 1);
        if (cfg.disable_tls_scratch)
            setenv("LLAMINAR_RMSNORM_DISABLE_TLS_SCRATCH", "1", 1);
        if (cfg.prealloc_rows > 0)
        {
            std::string v = std::to_string(cfg.prealloc_rows);
            setenv("LLAMINAR_RMSNORM_SCRATCH_PREALLOC_ROWS", v.c_str(), 1);
        }
        // Refresh snapshot so core honors flags
        llaminar::debugEnvRefresh();

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> src(rows * cols), gamma(cols), dst(rows * cols);
        for (auto &v : src)
            v = dist(rng);
        for (auto &g : gamma)
            g = 0.5f + 0.5f * dist(rng);
        rmsnorm_row_major_fused(src.data(), gamma.data(), dst.data(), rows, cols, 1e-5f, GammaMode::REPLICATED, 0, {});
        return dst;
    }

    void parity_suite(size_t rows, size_t cols)
    {
        unsigned seed = 1234 + (unsigned)(rows * 17 + cols);
        // Baseline (default env)
        auto base = run_case(rows, cols, {}, seed);
        // Disable TLS scratch
        auto no_tls = run_case(rows, cols, {.force_scalar = false, .disable_tls_scratch = true, .prealloc_rows = 0}, seed);
        EXPECT_LT(rel_l2(base, no_tls), 1e-7) << "Mismatch: disable_tls_scratch parity failed (rows=" << rows << ", cols=" << cols << ")";
        // Force scalar (still with TLS scratch active)
        auto scalar = run_case(rows, cols, {.force_scalar = true, .disable_tls_scratch = false, .prealloc_rows = 0}, seed);
        EXPECT_LT(rel_l2(base, scalar), 1e-7) << "Mismatch: force_scalar parity failed (rows=" << rows << ", cols=" << cols << ")";
        // Preallocate (should be identical)
        auto prealloc = run_case(rows, cols, {.force_scalar = false, .disable_tls_scratch = false, .prealloc_rows = (int)rows}, seed);
        EXPECT_LT(rel_l2(base, prealloc), 1e-7) << "Mismatch: prealloc parity failed (rows=" << rows << ", cols=" << cols << ")";
        // Combined worst-case (scalar + no TLS)
        auto scalar_no_tls = run_case(rows, cols, {.force_scalar = true, .disable_tls_scratch = true, .prealloc_rows = 0}, seed);
        EXPECT_LT(rel_l2(base, scalar_no_tls), 1e-7) << "Mismatch: scalar+no_tls parity failed (rows=" << rows << ", cols=" << cols << ")";
        // Clean up env for other tests
        clear_rmsnorm_env();
        llaminar::debugEnvRefresh();
    }

    TEST(RMSNormCoreParity, SmallRowSingle)
    {
        parity_suite(1, 4096);
    }

    TEST(RMSNormCoreParity, MultiRowMedium)
    {
        parity_suite(16, 4096);
    }

    TEST(RMSNormCoreParity, MinimalEdge)
    {
        parity_suite(1, 1);
    }

} // anonymous namespace
