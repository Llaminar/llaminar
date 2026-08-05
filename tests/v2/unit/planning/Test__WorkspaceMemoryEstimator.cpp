/**
 * @file Test__WorkspaceMemoryEstimator.cpp
 * @brief Verifies dense, hybrid recurrent, and exact MoE workspace preflight sizing.
 *
 * These tests protect the boundary between metadata-only memory planning and
 * the exact graph-family allocator. In particular, a 4K Qwen35MoE graph must
 * reserve enough memory before weight loading for the production CUDA/ROCm
 * requirement factories to publish all stable workspace names.
 */

#include <gtest/gtest.h>
#include "planning/WorkspaceMemoryEstimator.h"
#include "backends/DeviceId.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include <stdexcept>

using namespace llaminar2;

namespace
{

/**
 * @brief Construct the routing geometry of the production Qwen3.6 35B MoE model.
 * @param hybrid Whether to include one GDN layer's projection geometry.
 * @return Metadata-only profile suitable for workspace estimation.
 */
ModelMemoryProfile qwen35MoEProfile(bool hybrid)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 41;
    profile.d_model = 2048;
    profile.d_ff = 512;
    profile.n_heads = 16;
    profile.n_kv_heads = 2;
    profile.head_dim = 128;
    profile.vocab_size = 248320;
    profile.max_seq_len = 4096;
    profile.expert_count = 256;
    profile.expert_used_count = 8;
    profile.expert_feed_forward_length = 512;
    profile.expert_shared_feed_forward_length = 512;

    if (hybrid)
    {
        TensorSizeInfo recurrent_marker;
        recurrent_marker.name = "blk.0.ssm_out.weight";
        recurrent_marker.layer_index = 0;
        profile.tensors.push_back(std::move(recurrent_marker));

        TensorSizeInfo qkv;
        qkv.name = "blk.0.attn_qkv.weight";
        qkv.elements = size_t{8192} * size_t{2048};
        qkv.K = 2048;
        qkv.layer_index = 0;
        profile.tensors.push_back(std::move(qkv));

        TensorSizeInfo gate;
        gate.name = "blk.0.attn_gate.weight";
        gate.elements = size_t{4096} * size_t{2048};
        gate.K = 2048;
        gate.layer_index = 0;
        profile.tensors.push_back(std::move(gate));
    }
    return profile;
}

} // namespace

TEST(Test__WorkspaceMemoryEstimator, GPU_ReturnsNonZero)
{
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes, 0u);
    EXPECT_GE(bytes, 800ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, CPU_ReturnsZero)
{
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cpu());

    EXPECT_EQ(bytes, 0u);
}

TEST(Test__WorkspaceMemoryEstimator, GPU_HasMinimumFloor)
{
    // Even a tiny model publishes the complete non-MoE graph-family envelope.
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 128, 64, 256, 100, DeviceId::cuda(0));

    EXPECT_GE(bytes, 800ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, CUDA_AddsExactMoERequirementFactoryBytes)
{
    auto moe = qwen35MoEProfile(false);
    auto dense = moe;
    dense.expert_count = 0;
    dense.expert_used_count = 0;
    dense.expert_feed_forward_length = 0;
    dense.expert_shared_feed_forward_length = 0;

    const size_t dense_bytes = WorkspaceMemoryEstimator::estimate(
        dense, 1, 4096, 512, 0, 40, 1, DeviceId::cuda(0));
    const size_t moe_bytes = WorkspaceMemoryEstimator::estimate(
        moe, 1, 4096, 512, 0, 40, 1, DeviceId::cuda(0));
    const size_t exact_moe_bytes =
        MoEWorkspaceBuffers::cudaMoE(4096, 2048, 512, 256, 8)
            .total_bytes_with_alignment();

    EXPECT_EQ(moe_bytes - dense_bytes, exact_moe_bytes);
}

TEST(Test__WorkspaceMemoryEstimator, ROCm_AddsExactMoERequirementFactoryBytes)
{
    auto moe = qwen35MoEProfile(false);
    auto dense = moe;
    dense.expert_count = 0;
    dense.expert_used_count = 0;
    dense.expert_feed_forward_length = 0;
    dense.expert_shared_feed_forward_length = 0;

    const size_t dense_bytes = WorkspaceMemoryEstimator::estimate(
        dense, 1, 4096, 512, 0, 40, 1, DeviceId::rocm(0));
    const size_t moe_bytes = WorkspaceMemoryEstimator::estimate(
        moe, 1, 4096, 512, 0, 40, 1, DeviceId::rocm(0));
    const size_t exact_moe_bytes =
        MoEWorkspaceBuffers::rocmMoE(4096, 2048, 512, 256, 8)
            .total_bytes_with_alignment();

    EXPECT_EQ(moe_bytes - dense_bytes, exact_moe_bytes);
}

TEST(Test__WorkspaceMemoryEstimator, DeclaredMoERejectsIncompleteGeometry)
{
    auto profile = qwen35MoEProfile(false);
    profile.expert_used_count = 0;

    EXPECT_THROW(
        WorkspaceMemoryEstimator::estimate(
            profile, 1, 4096, 512, 0, 40, 1, DeviceId::cuda(0)),
        std::runtime_error);
}

TEST(Test__WorkspaceMemoryEstimator, Qwen35MoE4K_CoversObservedCUDAFamilyPlan)
{
    const auto profile = qwen35MoEProfile(true);
    const size_t bytes = WorkspaceMemoryEstimator::estimate(
        profile,
        1,
        4096,
        512,
        0,
        40,
        1,
        DeviceId::cuda(0));

    constexpr size_t kObservedExactFamilyBytes = 1801622276ULL;
    EXPECT_GE(bytes, kObservedExactFamilyBytes)
        << "Preflight must reserve at least the exact production graph-family "
           "plan measured for Qwen3.6-35B-A3B at 4K rows.";
}

TEST(Test__WorkspaceMemoryEstimator, PreparedEmbeddingPathDoesNotReserveEmbeddingTableTemp)
{
    // Qwen3.6-scale vocab and hidden dimensions should not force a
    // vocab×d_model embedding-table workspace reservation in production.
    // The embedding kernel declares that fallback buffer only when prepared
    // embedding weights are unavailable.
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 5120, 27648, 151936, DeviceId::cuda(0));

    const size_t embedding_table_temp =
        size_t{151936} * size_t{5120} * sizeof(float);
    EXPECT_GT(bytes, 768ULL * 1024 * 1024)
        << "grouped-prefill accumulators must scale with resident rows";
    EXPECT_LT(bytes, embedding_table_temp)
        << "prepared embedding must not reserve a vocab-by-hidden fallback table";
}

TEST(Test__WorkspaceMemoryEstimator, VeryLongContextCanExceedFloorFromGemmOverhead)
{
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 32768, 5120, 27648, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes, 768ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, ROCm_SameAsGPU)
{
    size_t cuda_bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cuda(0));
    size_t rocm_bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::rocm(0));

    EXPECT_EQ(cuda_bytes, rocm_bytes);
}
