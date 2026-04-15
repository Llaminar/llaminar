/**
 * @file Test__DeviceWeightPreparer.cpp
 * @brief Unit tests for DeviceWeightPreparer — per-stage weight preparation for PP
 *
 * Tests the DeviceWeightPreparer component that takes a WeightViewSet and prepares
 * all contained weights for execution on a target device:
 *   - GEMM weights → KernelFactory::getOrCreatePreparedGemmWeights()
 *   - Non-GEMM weights → ITensor::ensureOnDevice()
 *
 * All tests use CPU target (DeviceId::cpu()) to avoid GPU dependencies.
 */

#include <gtest/gtest.h>
#include "kernels/DeviceWeightPreparer.h"
#include "loaders/WeightViewSet.h"
#include "backends/DeviceId.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test fixture
// =============================================================================

class Test__DeviceWeightPreparer : public ::testing::Test
{
protected:
    DeviceWeightPreparer preparer_;
    DeviceId cpu_device_ = DeviceId::cpu();
};

// =============================================================================
// Empty input
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, EmptyViewSetReturnsZeroCounts)
{
    WeightViewSet views(0, 4, true, false);
    // No views added → empty iteration

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 0u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 0u);
    EXPECT_EQ(result.total_device_bytes, 0u);
    EXPECT_EQ(result.failures, 0u);
    EXPECT_TRUE(result.failed_names.empty());
}

// =============================================================================
// GEMM weight preparation (CPU target → VNNI repacking)
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, SingleGemmWeightOnCPU)
{
    auto tensor = TestTensorFactory::createQ8_0Random({64, 64});

    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_q.weight", std::move(tensor), 0, /*is_gemm=*/true});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 1u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 0u);
    EXPECT_GT(result.total_device_bytes, 0u);
    EXPECT_EQ(result.failures, 0u);
}

TEST_F(Test__DeviceWeightPreparer, MultipleGemmWeightsOnCPU)
{
    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_q.weight", TestTensorFactory::createQ8_0Random({64, 64}), 0, true});
    views.addView({"blk.0.attn_k.weight", TestTensorFactory::createQ8_0Random({32, 64}), 0, true});
    views.addView({"blk.0.attn_v.weight", TestTensorFactory::createQ8_0Random({32, 64}), 0, true});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 3u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 0u);
    EXPECT_EQ(result.failures, 0u);
}

// =============================================================================
// Non-GEMM weight preparation (CPU target → no-op)
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, SingleNonGemmWeightOnCPU)
{
    auto tensor = TestTensorFactory::createFP32Random({64});

    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_norm.weight", std::move(tensor), 0, /*is_gemm=*/false});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 0u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 1u);
    EXPECT_GT(result.total_device_bytes, 0u);
    EXPECT_EQ(result.failures, 0u);
}

TEST_F(Test__DeviceWeightPreparer, MultipleNonGemmWeightsOnCPU)
{
    WeightViewSet views(0, 2, true, false);
    views.addView({"token_embd.weight", TestTensorFactory::createFP32Random({100, 64}), -1, false});
    views.addView({"blk.0.attn_norm.weight", TestTensorFactory::createFP32Random({64}), 0, false});
    views.addView({"blk.0.ffn_norm.weight", TestTensorFactory::createFP32Random({64}), 0, false});
    views.addView({"blk.1.attn_norm.weight", TestTensorFactory::createFP32Random({64}), 1, false});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 0u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 4u);
    EXPECT_EQ(result.failures, 0u);
}

// =============================================================================
// Mixed GEMM + non-GEMM weights
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, MixedGemmAndNonGemmWeights)
{
    WeightViewSet views(0, 1, true, false);
    // Embedding (non-GEMM)
    views.addView({"token_embd.weight", TestTensorFactory::createFP32Random({100, 64}), -1, false});
    // Norm (non-GEMM)
    views.addView({"blk.0.attn_norm.weight", TestTensorFactory::createFP32Random({64}), 0, false});
    // GEMM weights
    views.addView({"blk.0.attn_q.weight", TestTensorFactory::createQ8_0Random({64, 64}), 0, true});
    views.addView({"blk.0.ffn_gate.weight", TestTensorFactory::createQ8_0Random({128, 64}), 0, true});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 2u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 2u);
    EXPECT_GT(result.total_device_bytes, 0u);
    EXPECT_EQ(result.failures, 0u);
}

// =============================================================================
// Null tensor handling
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, NullTensorCountsAsFailure)
{
    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.broken.weight", nullptr, 0, true});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.failures, 1u);
    ASSERT_EQ(result.failed_names.size(), 1u);
    EXPECT_EQ(result.failed_names[0], "blk.0.broken.weight");
    EXPECT_EQ(result.gemm_weights_prepared, 0u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 0u);
}

TEST_F(Test__DeviceWeightPreparer, NullTensorAmongValidWeights)
{
    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_q.weight", TestTensorFactory::createQ8_0Random({64, 64}), 0, true});
    views.addView({"blk.0.broken.weight", nullptr, 0, false});
    views.addView({"blk.0.attn_norm.weight", TestTensorFactory::createFP32Random({64}), 0, false});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 1u);
    EXPECT_EQ(result.non_gemm_weights_uploaded, 1u);
    EXPECT_EQ(result.failures, 1u);
    ASSERT_EQ(result.failed_names.size(), 1u);
    EXPECT_EQ(result.failed_names[0], "blk.0.broken.weight");
}

// =============================================================================
// Result struct sanity
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, TotalDeviceBytesAccumulates)
{
    // Create two tensors with known sizes
    auto q_weight = TestTensorFactory::createQ8_0Random({128, 64});
    auto norm = TestTensorFactory::createFP32Random({64});

    auto q_bytes = q_weight->size_bytes();
    auto norm_bytes = norm->size_bytes();
    ASSERT_GT(q_bytes, 0u);
    ASSERT_GT(norm_bytes, 0u);

    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_q.weight", std::move(q_weight), 0, true});
    views.addView({"blk.0.attn_norm.weight", std::move(norm), 0, false});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.total_device_bytes, q_bytes + norm_bytes);
}

// =============================================================================
// Idempotency — double prepare should succeed (cache hits)
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, DoublePrepareIsIdempotent)
{
    auto tensor = TestTensorFactory::createQ8_0Random({64, 64});

    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_q.weight", std::move(tensor), 0, true});

    auto result1 = preparer_.prepare(views, cpu_device_);
    ASSERT_EQ(result1.failures, 0u);

    // Second prepare should be a cache hit in KernelFactory
    auto result2 = preparer_.prepare(views, cpu_device_);
    EXPECT_EQ(result2.failures, 0u);
    EXPECT_EQ(result2.gemm_weights_prepared, 1u);
}

// =============================================================================
// Different quantization formats
// =============================================================================

TEST_F(Test__DeviceWeightPreparer, Q4_0GemmWeight)
{
    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.ffn_down.weight", TestTensorFactory::createQ4_0Random({64, 64}), 0, true});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 1u);
    EXPECT_EQ(result.failures, 0u);
}

TEST_F(Test__DeviceWeightPreparer, MultiFormatGemmWeights)
{
    WeightViewSet views(0, 1, false, false);
    views.addView({"blk.0.attn_q.weight", TestTensorFactory::createQ8_0Random({64, 64}), 0, true});
    views.addView({"blk.0.ffn_gate.weight", TestTensorFactory::createQ4_0Random({128, 64}), 0, true});

    auto result = preparer_.prepare(views, cpu_device_);

    EXPECT_EQ(result.gemm_weights_prepared, 2u);
    EXPECT_EQ(result.failures, 0u);
}
