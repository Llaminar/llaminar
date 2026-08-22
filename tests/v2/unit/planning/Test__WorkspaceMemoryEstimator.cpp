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

/** @brief Build complete physical geometry for one device-free planner test. */
WorkspaceMemoryGeometry graphGeometry(
    DeviceId device,
    int local_d_ff = 512,
    int total_shards = 1)
{
    return WorkspaceMemoryGeometry{
        .device = device,
        .device_compute_units = device.is_cuda() ? 82 : 60,
        .batch_size = 1,
        .resident_graph_rows = 4096,
        .max_context_rows = 4096,
        .local_d_ff = local_d_ff,
        .first_layer = 0,
        .last_layer = 40,
        .total_shards = total_shards,
        .apportioned_routed_experts = false,
    };
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
        dense, graphGeometry(DeviceId::cuda(0)));
    const size_t moe_bytes = WorkspaceMemoryEstimator::estimate(
        moe, graphGeometry(DeviceId::cuda(0)));
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
        dense, graphGeometry(DeviceId::rocm(0)));
    const size_t moe_bytes = WorkspaceMemoryEstimator::estimate(
        moe, graphGeometry(DeviceId::rocm(0)));
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
            profile, graphGeometry(DeviceId::cuda(0))),
        std::runtime_error);
}

/**
 * @brief The 122B follower must price the direct mapped graph, not a compact surrogate.
 *
 * At 768 rows the two ROCm formulas differ by roughly 100 MiB. The old
 * compact-only estimate admitted a graph after expert weights had consumed
 * the memory needed by its real `(rows, top_k)` workspace, causing a late
 * allocation failure during participant graph construction.
 */
TEST(Test__WorkspaceMemoryEstimator,
     ROCmRoutedParticipantMergesMappedAndCompactEndpointShapes)
{
    auto profile = qwen35MoEProfile(false);
    profile.d_model = 3072;
    profile.expert_feed_forward_length = 1024;

    auto geometry = graphGeometry(DeviceId::rocm(0));
    geometry.resident_graph_rows = 768;
    geometry.max_context_rows = 4096;
    geometry.apportioned_routed_experts = true;

    WorkspaceRequirements expected = MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/768,
        /*d_model=*/3072,
        /*intermediate=*/1024,
        /*num_experts=*/256,
        /*top_k=*/8);
    expected.merge(MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/768 * 8,
        /*d_model=*/3072,
        /*intermediate=*/1024,
        /*num_experts=*/256,
        /*top_k=*/1));

    const size_t compact_only =
        MoEWorkspaceBuffers::rocmMoE(768 * 8, 3072, 1024, 256, 1)
            .total_bytes_with_alignment();
    const size_t actual =
        WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
            profile, geometry);
    EXPECT_EQ(actual, expected.total_bytes_with_alignment());
    EXPECT_GT(actual, compact_only + 90ULL * 1024ULL * 1024ULL);
}

TEST(Test__WorkspaceMemoryEstimator, Qwen35MoE4K_CoversObservedCUDAFamilyPlan)
{
    const auto profile = qwen35MoEProfile(true);
    const size_t bytes = WorkspaceMemoryEstimator::estimate(
        profile,
        graphGeometry(DeviceId::cuda(0)));

    constexpr size_t kObservedExactFamilyBytes = 1801622276ULL;
    EXPECT_GE(bytes, kObservedExactFamilyBytes)
        << "Preflight must reserve at least the exact production graph-family "
           "plan measured for Qwen3.6-35B-A3B at 4K rows.";
}

TEST(Test__WorkspaceMemoryEstimator,
     Qwen122TP2_CoversExactAttentionAndFloatingProjectionArenas)
{
    auto profile = qwen35MoEProfile(true);
    profile.architecture = "qwen35moe";
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.expert_feed_forward_length = 1024;
    profile.full_attention_interval = 4;

    TensorSizeInfo full_attention_q_gate;
    full_attention_q_gate.name = "blk.3.attn_q.weight";
    full_attention_q_gate.quant_type = "BF16";
    full_attention_q_gate.elements = size_t{16384} * size_t{3072};
    full_attention_q_gate.K = 3072;
    full_attention_q_gate.layer_index = 3;
    profile.tensors.push_back(std::move(full_attention_q_gate));

    TensorSizeInfo gdn_qkv;
    gdn_qkv.name = "blk.0.attn_qkv.weight";
    gdn_qkv.quant_type = "BF16";
    gdn_qkv.elements = size_t{12288} * size_t{3072};
    gdn_qkv.K = 3072;
    gdn_qkv.layer_index = 0;
    profile.tensors.push_back(std::move(gdn_qkv));

    TensorSizeInfo gdn_gate;
    gdn_gate.name = "blk.0.attn_gate.weight";
    gdn_gate.quant_type = "BF16";
    gdn_gate.elements = size_t{8192} * size_t{3072};
    gdn_gate.K = 3072;
    gdn_gate.layer_index = 0;
    profile.tensors.push_back(std::move(gdn_gate));

    const size_t bytes = WorkspaceMemoryEstimator::estimate(
        profile,
        graphGeometry(DeviceId::cuda(0), /*local_d_ff=*/512,
                      /*total_shards=*/2));

    constexpr size_t kObservedExactFamilyBytes = 4232756740ULL;
    EXPECT_GE(bytes, kObservedExactFamilyBytes)
        << "The 122B TP2 preflight must cover CUDA FA2 context summaries, "
           "the eight-way floating projection redirect, and the existing MoE/GDN family.";
}

TEST(Test__WorkspaceMemoryEstimator,
     Qwen122TP2Depth15CoversRetainedMTPSerialFamily)
{
    auto profile = qwen35MoEProfile(true);
    profile.architecture = "qwen35moe";
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.expert_feed_forward_length = 1024;
    profile.full_attention_interval = 4;

    TensorSizeInfo full_attention_q_gate;
    full_attention_q_gate.name = "blk.3.attn_q.weight";
    full_attention_q_gate.quant_type = "BF16";
    full_attention_q_gate.elements = size_t{16384} * size_t{3072};
    full_attention_q_gate.K = 3072;
    full_attention_q_gate.layer_index = 3;
    profile.tensors.push_back(std::move(full_attention_q_gate));

    auto geometry = graphGeometry(
        DeviceId::cuda(0), /*local_d_ff=*/512, /*total_shards=*/2);
    geometry.resident_graph_rows = 16;
    geometry.mtp_target_query_rows = 16;

    const size_t bytes = WorkspaceMemoryEstimator::estimate(
        profile, geometry);
    constexpr size_t kObservedExactDepth15SerialFamilyBytes =
        2288657668ULL;
    EXPECT_GE(bytes, kObservedExactDepth15SerialFamilyBytes)
        << "Depth-fifteen admission must price the retained main, grouped-verifier, and MTP namespaces before loading experts.";
}

TEST(Test__WorkspaceMemoryEstimator,
     CUDAFloatingProjectionRedirectUsesLocalOutputShard)
{
    auto floating = qwen35MoEProfile(false);
    floating.n_heads = 32;
    floating.head_dim = 256;

    TensorSizeInfo q_gate;
    q_gate.name = "blk.0.attn_q.weight";
    q_gate.quant_type = "BF16";
    q_gate.elements = size_t{16384} * size_t{3072};
    q_gate.K = 3072;
    q_gate.layer_index = 0;
    floating.tensors.push_back(q_gate);

    auto quantized = floating;
    quantized.tensors.back().quant_type = "Q8_0";
    const auto geometry = graphGeometry(
        DeviceId::cuda(0), /*local_d_ff=*/512, /*total_shards=*/2);
    const size_t floating_bytes =
        WorkspaceMemoryEstimator::estimate(floating, geometry);
    const size_t quantized_bytes =
        WorkspaceMemoryEstimator::estimate(quantized, geometry);

    constexpr size_t kLocalProjectionColumns = 8192;
    constexpr size_t kRedirectBytes =
        8ULL * 4096ULL * kLocalProjectionColumns * sizeof(float);
    constexpr size_t kThreeAlignedPointerArrays = 3ULL * 256ULL;
    EXPECT_EQ(
        floating_bytes - quantized_bytes,
        kRedirectBytes + kThreeAlignedPointerArrays);
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
