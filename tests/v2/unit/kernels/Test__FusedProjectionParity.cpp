/**
 * @file Test__FusedProjectionParity.cpp
 * @brief Regression tests for multiply_fused_tensor vs multiply_tensor parity.
 *
 * These tests verify that the fused multi-projection path
 * (multiply_fused_tensor) produces identical results to calling
 * multiply_tensor independently for each projection.
 *
 * Regression test for: Q4_0 fused GEMV (M=1) producing incorrect output
 * due to compiler miscompilation of inlined AVX-512 VNNI code at -O3.
 * The bug only manifested in the fused path for M=1 decode with Q4_0
 * weights, producing max_diff > 100 compared to multiply_tensor output.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test fixture
// ============================================================================

class FusedProjectionParity : public ::testing::Test
{
protected:
    static constexpr float PARITY_TOLERANCE = 1e-5f;

    struct ProjectionSpec
    {
        int n;
        const char *name;
    };

    /// Create a quantized weight tensor of the given type with the given shape.
    /// Supported: "Q4_0", "Q8_0", "IQ4_NL"
    static std::unique_ptr<TensorBase> createWeights(
        const std::string &type, size_t n, size_t k, uint32_t seed)
    {
        std::vector<size_t> shape = {n, k};
        if (type == "Q4_0")
            return TestTensorFactory::createQ4_0Random(shape, seed);
        if (type == "Q8_0")
            return TestTensorFactory::createQ8_0Random(shape, seed);
        if (type == "IQ4_NL")
            return TestTensorFactory::createIQ4_NLRandom(shape, seed);
        return nullptr;
    }

    /// Run parity test: multiply_fused_tensor vs per-projection multiply_tensor.
    /// Returns max absolute difference across all projections.
    static float runFusedVsSequentialParity(
        const std::string &weight_type,
        const std::vector<ProjectionSpec> &proj_specs,
        int m, int k,
        unsigned seed = 42)
    {
        // Create input
        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(m), static_cast<size_t>(k)}, -1.0f, 1.0f, seed);

        // Create weight tensors and kernels for each projection
        struct ProjData
        {
            std::unique_ptr<TensorBase> weights;
            std::unique_ptr<ITensorGemm> kernel;
            std::unique_ptr<FP32Tensor> output_fused;
            std::unique_ptr<FP32Tensor> output_sequential;
        };

        std::vector<ProjData> projs;
        projs.reserve(proj_specs.size());

        for (size_t i = 0; i < proj_specs.size(); ++i)
        {
            ProjData pd;
            int n = proj_specs[i].n;

            pd.weights = createWeights(weight_type,
                                       static_cast<size_t>(n),
                                       static_cast<size_t>(k),
                                       seed + 100 + static_cast<uint32_t>(i));
            EXPECT_NE(pd.weights, nullptr) << "Failed to create " << weight_type << " weights";
            if (!pd.weights)
                return 1e10f;

            pd.kernel = pd.weights->createGemm();
            EXPECT_NE(pd.kernel, nullptr) << "Failed to create GEMM kernel";
            if (!pd.kernel)
                return 1e10f;

            pd.output_fused = TestTensorFactory::createFP32Zeros(
                {static_cast<size_t>(m), static_cast<size_t>(n)});
            pd.output_sequential = TestTensorFactory::createFP32Zeros(
                {static_cast<size_t>(m), static_cast<size_t>(n)});

            projs.push_back(std::move(pd));
        }

        // --- Path 1: multiply_fused_tensor ---
        {
            std::vector<ITensorGemm::TensorProjectionDesc> fused_descs;
            for (size_t i = 0; i < proj_specs.size(); ++i)
            {
                fused_descs.emplace_back(
                    projs[i].kernel.get(),
                    projs[i].output_fused.get(),
                    proj_specs[i].n,
                    nullptr,
                    proj_specs[i].name);
            }

            bool fused_ok = projs[0].kernel->multiply_fused_tensor(
                input.get(), fused_descs, m, k);
            EXPECT_TRUE(fused_ok) << "multiply_fused_tensor failed";
            if (!fused_ok)
                return 1e10f;
        }

        // --- Path 2: multiply_tensor per projection ---
        for (size_t i = 0; i < proj_specs.size(); ++i)
        {
            bool seq_ok = projs[i].kernel->multiply_tensor(
                input.get(), projs[i].output_sequential.get(),
                m, proj_specs[i].n, k);
            EXPECT_TRUE(seq_ok) << "multiply_tensor failed for projection " << i;
            if (!seq_ok)
                return 1e10f;
        }

        // --- Compare outputs ---
        float global_max_diff = 0.0f;
        for (size_t i = 0; i < proj_specs.size(); ++i)
        {
            int n = proj_specs[i].n;
            const float *fused = projs[i].output_fused->data();
            const float *seq = projs[i].output_sequential->data();
            size_t count = static_cast<size_t>(m) * n;

            int mismatches = 0;
            float max_diff = 0.0f;
            for (size_t j = 0; j < count; ++j)
            {
                float diff = std::abs(fused[j] - seq[j]);
                if (diff > PARITY_TOLERANCE)
                    ++mismatches;
                max_diff = std::max(max_diff, diff);
            }

            if (mismatches > 0)
            {
                std::cout << "[MISMATCH] proj=" << proj_specs[i].name
                          << " N=" << n
                          << " mismatches=" << mismatches << "/" << count
                          << " max_diff=" << max_diff
                          << " fused[0]=" << fused[0]
                          << " seq[0]=" << seq[0] << std::endl;
            }

            global_max_diff = std::max(global_max_diff, max_diff);
        }

        return global_max_diff;
    }
};

// ============================================================================
// Q4_0 fused projection parity (regression test for M=1 decode bug)
// ============================================================================

// QKV layout: Q[896], K[128], V[128] — matches Qwen2.5-0.5B dimensions
TEST_F(FusedProjectionParity, Q4_0_FusedQKV_M1_Decode)
{
    constexpr int K = 896;

    float max_diff = runFusedVsSequentialParity(
        "Q4_0",
        {{896, "Q_proj"}, {128, "K_proj"}, {128, "V_proj"}},
        1, K);

    EXPECT_LT(max_diff, PARITY_TOLERANCE)
        << "Q4_0 fused QKV (M=1) diverges from sequential";
}

// Gate/Up layout: gate[4864], up[4864] — matches Qwen2.5-0.5B FFN dimensions
TEST_F(FusedProjectionParity, Q4_0_FusedGateUp_M1_Decode)
{
    constexpr int K = 896;

    float max_diff = runFusedVsSequentialParity(
        "Q4_0",
        {{4864, "gate"}, {4864, "up"}},
        1, K);

    EXPECT_LT(max_diff, PARITY_TOLERANCE)
        << "Q4_0 fused gate/up (M=1) diverges from sequential";
}

// M>1 prefill path (should always work, sanity check)
TEST_F(FusedProjectionParity, Q4_0_FusedQKV_M64_Prefill)
{
    constexpr int K = 896;

    float max_diff = runFusedVsSequentialParity(
        "Q4_0",
        {{896, "Q_proj"}, {128, "K_proj"}, {128, "V_proj"}},
        64, K);

    EXPECT_LT(max_diff, PARITY_TOLERANCE)
        << "Q4_0 fused QKV (M=64) diverges from sequential";
}

// ============================================================================
// Q8_0 fused projection parity (was never broken, but verify consistency)
// ============================================================================

TEST_F(FusedProjectionParity, Q8_0_FusedQKV_M1_Decode)
{
    constexpr int K = 896;

    float max_diff = runFusedVsSequentialParity(
        "Q8_0",
        {{896, "Q_proj"}, {128, "K_proj"}, {128, "V_proj"}},
        1, K);

    EXPECT_LT(max_diff, PARITY_TOLERANCE)
        << "Q8_0 fused QKV (M=1) diverges from sequential";
}

// ============================================================================
// IQ4_NL fused projection parity (nibble-LUT format, same code path as Q4_0)
// ============================================================================

TEST_F(FusedProjectionParity, IQ4_NL_FusedQKV_M1_Decode)
{
    constexpr int K = 896;

    float max_diff = runFusedVsSequentialParity(
        "IQ4_NL",
        {{896, "Q_proj"}, {128, "K_proj"}, {128, "V_proj"}},
        1, K);

    EXPECT_LT(max_diff, PARITY_TOLERANCE)
        << "IQ4_NL fused QKV (M=1) diverges from sequential";
}

// ============================================================================
// Edge case: single projection (fused with 1 element = degenerate case)
// ============================================================================

TEST_F(FusedProjectionParity, Q4_0_SingleProjection_M1)
{
    constexpr int K = 896;

    float max_diff = runFusedVsSequentialParity(
        "Q4_0",
        {{512, "single"}},
        1, K);

    EXPECT_LT(max_diff, PARITY_TOLERANCE)
        << "Q4_0 single projection (M=1) diverges from sequential";
}
