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
#include "config/GDNHeadAssignment.h"
#include "execution/compute_stages/stages/GDNSpeculativeWorkspaceContract.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "kernels/common/EmbeddingWorkspaceContract.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include "kernels/kvcache/KVCacheWorkspaceContract.h"
#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmWorkspaceContract.h"
#include "kernels/rope/RoPEWorkspaceContract.h"
#include "tensors/TensorType.h"
#include <algorithm>
#include <array>
#include <stdexcept>

using namespace llaminar2;

namespace
{

/**
 * @brief Price only the attention ABI when updating historical family goldens.
 * @param geometry Unchanged captured participant geometry.
 * @param heads Concrete local query heads.
 * @param kv_heads Concrete local KV heads.
 * @param head_dim Width of each local head.
 * @param family_rows Query range covered by the prefill envelope.
 * @return Canonical descriptor bytes, including compact and conversion storage.
 *
 * Historical runtime snapshots included summaries for unadmitted query widths.
 * Subtract only that named-buffer delta from those goldens: every other byte,
 * alignment check, and upper bound remains protected. This is test evidence,
 * not a second production allocation calculation.
 */
size_t rocmAttentionDescriptorBytes(
    const WorkspaceMemoryGeometry &geometry,
    int heads, int kv_heads, int head_dim, int family_rows)
{
    const auto family = rocm::fa2_policy::selectROCmFA2GeometrySelectedWorkspaceEnvelope({
        .batch_size = geometry.batch_size,
        .query_rows = family_rows,
        .local_query_heads = heads,
        .head_dim = head_dim,
        .kv_capacity = geometry.max_context_rows,
        .compute_unit_count = geometry.device_compute_units,
        .lds_capacity_bytes = rocm::fa2_policy::kROCmFA2LDSCapacityBytes,
        .requested_axis = attention::AttentionPrefillParallelAxis::GeometrySelected,
    });
    const auto cardinality = attention::planAttentionWorkspaceCardinality(
        geometry.resident_graph_rows, geometry.batch_size);
    return attention_workspace::requirements({
        .compact_query_rows = cardinality.compact_query_rows,
        .request_count = cardinality.request_count,
        .local_query_heads = heads,
        .local_kv_heads = kv_heads,
        .head_dim = head_dim,
        .context_rows = geometry.max_context_rows,
        .partial_output_floor_bytes = family.partial_output_bytes,
        .partial_m_floor_bytes = family.partial_m_bytes,
        .partial_l_floor_bytes = family.partial_l_bytes,
    }).total_bytes_with_alignment();
}

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
        profile.gdn_conv_kernel_size = 4;
        profile.gdn_state_size = 128;
        profile.gdn_inner_size = 4096;
        profile.gdn_group_count = 16;
        profile.gdn_time_step_rank = 32;

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

/**
 * @brief Expand a small hybrid profile to the 122B model's 36 main GDN layers.
 *
 * Qwen3.5 122B has 48 main blocks with every fourth block using full
 * attention, plus one trailing MTP block in the model inventory. The grouped
 * verifier retains rollback slots for the 36 recurrent main blocks only.
 */
void installQwen122HybridLayerInventory(ModelMemoryProfile& profile)
{
    profile.n_layers = 49;
    profile.full_attention_interval = 4;
    profile.gdn_conv_kernel_size = 4;
    profile.gdn_state_size = 128;
    profile.gdn_inner_size = 8192;
    profile.gdn_group_count = 16;
    profile.gdn_time_step_rank = 64;
    profile.tensors.erase(
        std::remove_if(
            profile.tensors.begin(),
            profile.tensors.end(),
            [](const TensorSizeInfo& tensor)
            {
                return tensor.name.ends_with(".ssm_out.weight");
            }),
        profile.tensors.end());
    for (int layer = 0; layer < 48; ++layer)
    {
        if ((layer + 1) % profile.full_attention_interval == 0)
            continue;
        TensorSizeInfo marker;
        marker.name = "blk." + std::to_string(layer) + ".ssm_out.weight";
        marker.layer_index = layer;
        profile.tensors.push_back(std::move(marker));
    }
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
    // The legacy dimension-only overload cannot express local attention heads
    // or the configured physical-core worker team. Production CPU planning
    // uses the model-aware overload exercised below.
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cpu());

    EXPECT_EQ(bytes, 0u);
}

/**
 * @brief Reproduce the exact mixed LocalPP CPU workspace admission failure.
 *
 * The CPU stage retains sixteen verifier rows. With fourteen query heads and
 * the production 28-physical-core OpenMP team, each head owns two producer
 * slots plus one deterministic merge slot. Runtime and metadata preflight must
 * therefore publish the same 177,408-byte stable-name family.
 */
TEST(Test__WorkspaceMemoryEstimator,
     Qwen2CPUStageUsesCanonicalParallelAttentionContract)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 24;
    profile.d_model = 896;
    profile.d_ff = 4864;
    profile.n_heads = 14;
    profile.n_kv_heads = 2;
    profile.head_dim = 64;
    profile.vocab_size = 151936;
    profile.max_seq_len = 4096;

    const WorkspaceMemoryGeometry geometry{
        .device = DeviceId::cpu(),
        .device_compute_units = 28,
        .batch_size = 1,
        .resident_graph_rows = 9,
        .max_context_rows = 4096,
        .local_d_ff = 4864,
        .local_query_heads = 14,
        .local_kv_heads = 2,
        .first_layer = 12,
        .last_layer = 23,
        .total_shards = 1,
    };
    const auto cardinality = attention::planAttentionWorkspaceCardinality(
        geometry.resident_graph_rows,
        geometry.batch_size);
    const auto runtime_contract =
        attention_workspace::cpuParallelRequirements({
            .compact_query_rows = cardinality.compact_query_rows,
            .local_query_heads = geometry.local_query_heads,
            .head_dim = profile.head_dim,
            .worker_count = geometry.device_compute_units,
        });

    ASSERT_NE(runtime_contract.find(attention_workspace::kPartialOutput), nullptr);
    ASSERT_NE(runtime_contract.find(attention_workspace::kPartialM), nullptr);
    ASSERT_NE(runtime_contract.find(attention_workspace::kPartialL), nullptr);
    EXPECT_EQ(runtime_contract.total_bytes_with_alignment(), 177408u);
    EXPECT_EQ(
        WorkspaceMemoryEstimator::estimate(profile, geometry),
        runtime_contract.total_bytes_with_alignment());
}

TEST(Test__WorkspaceMemoryEstimator, GPU_HasMinimumFloor)
{
    // Even a tiny model publishes the complete non-MoE graph-family envelope.
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 128, 64, 256, 100, DeviceId::cuda(0));

    EXPECT_GE(bytes, 800ULL * 1024 * 1024);
}

/** @brief Exact local head geometry admits a non-divisible rank-local TP plan. */
TEST(Test__WorkspaceMemoryEstimator, ExactLocalQueryHeadsPermitUnevenTP4)
{
    auto profile = qwen35MoEProfile(false);
    profile.n_heads = 14;
    profile.n_kv_heads = 2;
    profile.head_dim = 64;

    auto exact = graphGeometry(
        DeviceId::rocm(0), /*local_d_ff=*/128, /*total_shards=*/4);
    exact.local_query_heads = 4;
    EXPECT_GT(WorkspaceMemoryEstimator::estimate(profile, exact), 0u);

    auto lossy = exact;
    lossy.local_query_heads = 0;
    EXPECT_THROW(
        WorkspaceMemoryEstimator::estimate(profile, lossy),
        std::runtime_error);
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

/**
 * @brief Stable fused-scatter names are one physical buffer, not one per layer.
 *
 * Every quantized GDN block declares the same captured workspace ABI. Adding a
 * second block with identical geometry must therefore leave admission bytes
 * unchanged. This reproduces the metadata-only overcharge that previously
 * summed one `rocm_scatter_partial_batched` descriptor per model layer while
 * runtime correctly canonicalized the name.
 */
TEST(Test__WorkspaceMemoryEstimator,
     ROCmFusedProjectionWorkspaceCanonicalizesAcrossLayers)
{
    auto profile = qwen35MoEProfile(false);
    profile.expert_count = 0;
    profile.expert_used_count = 0;
    profile.expert_feed_forward_length = 0;
    profile.expert_shared_feed_forward_length = 0;

    const auto add_fused_gdn_bundle = [&](int layer)
    {
        const auto add_matrix = [&](std::string suffix,
                                    std::size_t output_columns)
        {
            TensorSizeInfo tensor;
            tensor.name = "blk." + std::to_string(layer) + suffix;
            tensor.quant_type = "Q8_K";
            tensor.elements = output_columns * size_t{2048};
            tensor.K = 2048;
            tensor.layer_index = layer;
            profile.tensors.push_back(std::move(tensor));
        };
        add_matrix(".attn_qkv.weight", 8192);
        add_matrix(".attn_gate.weight", 4096);
        add_matrix(".ssm_alpha.weight", 32);
        add_matrix(".ssm_beta.weight", 32);
    };

    add_fused_gdn_bundle(0);
    const size_t one_layer = WorkspaceMemoryEstimator::estimate(
        profile, graphGeometry(DeviceId::rocm(0)));

    add_fused_gdn_bundle(1);
    const size_t two_layers = WorkspaceMemoryEstimator::estimate(
        profile, graphGeometry(DeviceId::rocm(0)));

    EXPECT_EQ(two_layers, one_layer)
        << "A shared captured workspace name retains its maximum extent across "
           "serial layers; layer count cannot multiply physical bytes.";
}

/** @brief K/V conversion admission and runtime share one typed size policy. */
TEST(Test__WorkspaceMemoryEstimator,
     KVConversionContractRetainsConfiguredHorizonAndStablePair)
{
    const auto requirements =
        kv_cache_workspace::conversionRequirements({
            .configured_batch_size = 2,
            .configured_context_rows = 4096,
            .requested_graph_rows = 256,
            .requested_batch_size = 1,
            .conversion_row_bytes = 512,
            .native_row_bytes = 1024,
        });

    constexpr size_t kPerBufferBytes =
        size_t{2} * size_t{4096} * size_t{1024};
    ASSERT_EQ(requirements.buffers.size(), 2u);
    EXPECT_EQ(requirements.buffers[0].name,
              KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
    EXPECT_EQ(requirements.buffers[1].name,
              KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
    EXPECT_EQ(requirements.buffers[0].size_bytes, kPerBufferBytes);
    EXPECT_EQ(requirements.buffers[1].size_bytes, kPerBufferBytes);
    EXPECT_EQ(requirements.total_bytes_with_alignment(),
              2u * kPerBufferBytes);
}

/**
 * @brief Embedding ownership prices the exact retained token-ID descriptor.
 *
 * A nine-row LocalTP graph declares 36 bytes, aligned as one 256-byte serial
 * family member. Omitting that member previously left the production ROCm
 * interval plan four bytes beyond its admitted total after neighboring
 * descriptors reused the other 252 bytes of alignment padding.
 */
TEST(Test__WorkspaceMemoryEstimator,
     EmbeddingOwnerAddsCanonicalGraphStableTokenIdsABI)
{
    auto profile = qwen35MoEProfile(false);
    auto without_embedding = graphGeometry(DeviceId::rocm(0));
    without_embedding.resident_graph_rows = 9;
    without_embedding.owns_embedding = false;
    auto with_embedding = without_embedding;
    with_embedding.owns_embedding = true;

    const auto requirements = embedding_workspace::requirements({
        .graph_rows = 9,
    });
    ASSERT_EQ(requirements.buffers.size(), 1u);
    EXPECT_EQ(requirements.buffers.front().name,
              EmbeddingWorkspaceBuffers::TOKEN_IDS);
    EXPECT_EQ(requirements.buffers.front().size_bytes, 36u);
    EXPECT_EQ(requirements.total_bytes_with_alignment(), 256u);
    EXPECT_EQ(
        WorkspaceMemoryEstimator::estimate(profile, with_embedding) -
            WorkspaceMemoryEstimator::estimate(profile, without_embedding),
        requirements.total_bytes_with_alignment());
}

/**
 * @brief Reproduce the exact Qwen2 PP terminal-stage workspace admission miss.
 *
 * The historical serial family required 199,837,188 bytes. Its canonical
 * descriptor sum rounded to 199,837,440 bytes: quantized GEMM, full attention,
 * both independent K/V conversion pairs, and RoPE publications. Admission
 * formerly omitted the attention-owned pair and all RoPE state and reserved
 * only 195,633,920 bytes. Preserve that regression while deducting the exact
 * attention summaries retired by bounding query M to the admitted bucket.
 */
TEST(Test__WorkspaceMemoryEstimator,
     Qwen2ROCmPipelineStageCoversCanonicalAttentionAndRoPEABI)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 24;
    profile.d_model = 896;
    profile.d_ff = 4864;
    profile.n_heads = 14;
    profile.n_kv_heads = 2;
    profile.head_dim = 64;
    profile.vocab_size = 151936;
    profile.max_seq_len = 4096;

    const auto add_matrix = [&](std::string name,
                                int layer,
                                std::size_t output_columns,
                                std::size_t input_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = std::move(name);
        tensor.quant_type = "Q4_0";
        tensor.elements = output_columns * input_columns;
        tensor.K = input_columns;
        tensor.layer_index = layer;
        profile.tensors.push_back(std::move(tensor));
    };
    add_matrix("output.weight", -1, 151936, 896);
    for (int layer = 12; layer < 24; ++layer)
    {
        const std::string prefix = "blk." + std::to_string(layer);
        add_matrix(prefix + ".attn_q.weight", layer, 896, 896);
        add_matrix(prefix + ".attn_k.weight", layer, 128, 896);
        add_matrix(prefix + ".attn_v.weight", layer, 128, 896);
        add_matrix(prefix + ".attn_output.weight", layer, 896, 896);
        add_matrix(prefix + ".ffn_gate.weight", layer, 4864, 896);
        add_matrix(prefix + ".ffn_up.weight", layer, 4864, 896);
        add_matrix(prefix + ".ffn_down.weight", layer, 896, 4864);
    }

    const WorkspaceMemoryGeometry geometry{
        .device = DeviceId::rocm(1),
        .device_compute_units = 60,
        .batch_size = 1,
        .resident_graph_rows = 9,
        .max_context_rows = 4096,
        .local_d_ff = 4864,
        .local_query_heads = 14,
        .local_kv_heads = 2,
        .first_layer = 12,
        .last_layer = 23,
        .total_shards = 1,
    };

    const size_t removed_unadmitted_summaries =
        rocmAttentionDescriptorBytes(geometry, 14, 2, 64, geometry.max_context_rows) -
        rocmAttentionDescriptorBytes(geometry, 14, 2, 64, geometry.resident_graph_rows);
    const std::size_t runtime_serial_family_bytes = 199837188ULL - removed_unadmitted_summaries;
    const std::size_t aligned_canonical_admission_bytes = 199837440ULL - removed_unadmitted_summaries;
    const std::size_t admitted =
        WorkspaceMemoryEstimator::estimate(profile, geometry);
    EXPECT_EQ(admitted, aligned_canonical_admission_bytes);
    ASSERT_GE(admitted, runtime_serial_family_bytes);
    EXPECT_LT(admitted - runtime_serial_family_bytes, 256u)
        << "Only terminal descriptor alignment may exceed the runtime interval plan.";
}

/**
 * @brief A retained MTP sidecar is sized by verifier depth, not prefill rows.
 *
 * The sidecar projection runs only in the compact transaction. Varying the
 * unrelated main prefill bucket must therefore leave its physical workspace
 * unchanged. This catches the former 600-row charge for a three-row sidecar.
 */
TEST(Test__WorkspaceMemoryEstimator,
     ROCmRetainedMTPSidecarIgnoresMainPrefillBucket)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 49;
    profile.mtp_layer_count = 1;
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.vocab_size = 248320;
    profile.max_seq_len = 4096;

    TensorSizeInfo hybrid_marker;
    hybrid_marker.name = "blk.0.ssm_out.weight";
    hybrid_marker.layer_index = 0;
    profile.tensors.push_back(std::move(hybrid_marker));

    TensorSizeInfo sidecar;
    sidecar.name = "blk.48.nextn.eh_proj.weight";
    sidecar.quant_type = "Q8_K";
    sidecar.elements = size_t{3072} * size_t{6144};
    sidecar.K = 6144;
    sidecar.layer_index = 48;
    profile.tensors.push_back(std::move(sidecar));

    auto compact = graphGeometry(
        DeviceId::rocm(0), /*local_d_ff=*/256, /*total_shards=*/4);
    compact.first_layer = 48;
    compact.last_layer = 48;
    compact.local_query_heads = 8;
    compact.local_kv_heads = 2;
    compact.mtp_target_query_rows = 16;
    compact.resident_graph_rows = 32;

    auto large_prefill = compact;
    large_prefill.resident_graph_rows = 4096;

    EXPECT_EQ(
        WorkspaceMemoryEstimator::estimate(profile, compact),
        WorkspaceMemoryEstimator::estimate(profile, large_prefill));
}

/**
 * @brief Every production quantized codebook prices the same prepared ABI.
 *
 * Source codebooks change persistent packed-weight bytes, but ROCm prepares
 * every supported matrix into the same NativeVNNI execution contract. The
 * graph workspace therefore depends on N/K/rows and cannot vary by codebook.
 */
TEST(Test__WorkspaceMemoryEstimator,
     ROCmQuantizedWorkspaceIsCodebookIndependent)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 1;
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.vocab_size = 248320;
    profile.max_seq_len = 4096;

    TensorSizeInfo hybrid_marker;
    hybrid_marker.name = "blk.0.ssm_out.weight";
    hybrid_marker.layer_index = 0;
    profile.tensors.push_back(std::move(hybrid_marker));

    TensorSizeInfo projection;
    projection.name = "output.weight";
    projection.elements = size_t{248320} * size_t{3072};
    projection.K = 3072;
    projection.layer_index = -1;
    profile.tensors.push_back(std::move(projection));

    auto geometry = graphGeometry(
        DeviceId::rocm(0), /*local_d_ff=*/256, /*total_shards=*/4);
    geometry.resident_graph_rows = 16;
    geometry.local_query_heads = 8;
    geometry.local_kv_heads = 2;
    geometry.mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;

    size_t expected = 0u;
    size_t exercised_codebooks = 0u;
    for (int raw = 0; raw <= static_cast<int>(TensorType::AQ8); ++raw)
    {
        const auto type = static_cast<TensorType>(raw);
        if (!isNativeVnniFormat(type) && !isInt8VnniFormat(type))
            continue;
        profile.tensors.back().quant_type = tensorTypeName(type);
        const size_t actual =
            WorkspaceMemoryEstimator::estimate(profile, geometry);
        if (exercised_codebooks == 0u)
            expected = actual;
        EXPECT_EQ(actual, expected) << "codebook=" << tensorTypeName(type);
        ++exercised_codebooks;
    }
    EXPECT_EQ(exercised_codebooks, 21u)
        << "The canonical NativeVNNI predicates must enumerate every supported quantized codebook.";
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

    const auto add_expert_parent = [&](std::string suffix,
                                       std::size_t output_columns,
                                       std::size_t input_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = "blk.0." + std::move(suffix);
        tensor.quant_type = "Q4_K_XL";
        tensor.elements = output_columns * input_columns *
                          static_cast<std::size_t>(profile.expert_count);
        tensor.K = input_columns;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add_expert_parent("ffn_gate_exps.weight", 1024, 3072);
    add_expert_parent("ffn_up_exps.weight", 1024, 3072);
    add_expert_parent("ffn_down_exps.weight", 3072, 1024);

    WorkspaceRequirements direct = MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/768,
        /*d_model=*/3072,
        /*intermediate=*/1024,
        /*num_experts=*/256,
        /*top_k=*/8);
    direct.merge(
        rocm::quantized_gemm_workspace::projectionRequirements(
            /*rows=*/768, /*N=*/1024, /*K=*/3072));
    const std::vector<int> fused_columns(/*count=*/16, /*value=*/1024);
    rocm::quantized_gemm_workspace::appendFusedProjectionRequirements(
        direct, /*rows=*/768, fused_columns);

    WorkspaceRequirements compact = MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/768 * 8,
        /*d_model=*/3072,
        /*intermediate=*/1024,
        /*num_experts=*/256,
        /*top_k=*/1);
    compact.merge(
        rocm::quantized_gemm_workspace::projectionRequirements(
            /*rows=*/768 * 8, /*N=*/1024, /*K=*/3072));
    compact.merge(
        rocm::quantized_gemm_workspace::projectionRequirements(
            /*rows=*/768 * 8, /*N=*/3072, /*K=*/1024));

    direct.merge(compact);

    const size_t compact_only =
        MoEWorkspaceBuffers::rocmMoE(768 * 8, 3072, 1024, 256, 1)
            .total_bytes_with_alignment();
    const size_t actual =
        WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
            profile, geometry);
    EXPECT_EQ(actual, direct.total_bytes_with_alignment());
    EXPECT_GT(actual, compact_only + 90ULL * 1024ULL * 1024ULL);
}

/**
 * @brief Every prepared quantized expert codebook shares one ROCm GEMM ABI.
 *
 * Source compression changes persistent expert bytes, not the NativeVNNI
 * activation/reduction workspace. This sweep prevents a fix for the model's
 * current Q4_K_XL source format from becoming a format-specific admission
 * branch when future checkpoints select any other supported codebook.
 */
TEST(Test__WorkspaceMemoryEstimator,
     ROCmRoutedParticipantQuantizedWorkspaceIsCodebookIndependent)
{
    auto profile = qwen35MoEProfile(false);
    auto geometry = graphGeometry(DeviceId::rocm(0));
    geometry.resident_graph_rows = 9;
    geometry.apportioned_routed_experts = true;

    const auto add_expert_parent = [&](std::string suffix,
                                       std::size_t output_columns,
                                       std::size_t input_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = "blk.0." + std::move(suffix);
        tensor.elements = output_columns * input_columns *
                          static_cast<std::size_t>(profile.expert_count);
        tensor.K = input_columns;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add_expert_parent("ffn_gate_exps.weight", 512, 2048);
    add_expert_parent("ffn_up_exps.weight", 512, 2048);
    add_expert_parent("ffn_down_exps.weight", 2048, 512);

    std::size_t expected = 0u;
    std::size_t exercised_codebooks = 0u;
    for (int raw = 0; raw <= static_cast<int>(TensorType::AQ8); ++raw)
    {
        const TensorType type = static_cast<TensorType>(raw);
        if (!isNativeVnniFormat(type) && !isInt8VnniFormat(type))
            continue;
        for (TensorSizeInfo& tensor : profile.tensors)
            tensor.quant_type = tensorTypeName(type);

        const std::size_t actual =
            WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
                profile, geometry);
        if (exercised_codebooks == 0u)
            expected = actual;
        EXPECT_EQ(actual, expected) << "codebook=" << tensorTypeName(type);
        ++exercised_codebooks;
    }
    EXPECT_EQ(exercised_codebooks, 21u);
    EXPECT_EQ(expected, 374859776ULL)
        << "The nine-row Qwen3.6-35B mapped/compact follower family must "
           "retain the exact typed ROCm MoE and NativeVNNI workspace ABI.";
    EXPECT_GE(expected, 365474564ULL)
        << "Admission must cover the mapped 40-layer serial plan observed by "
           "the production graph allocator.";
}

/** @brief FP16, BF16, and FP32 routed experts publish the same ROCm pointer ABI. */
TEST(Test__WorkspaceMemoryEstimator,
     ROCmRoutedParticipantFloatingFormatsSharePointerWorkspaceABI)
{
    auto profile = qwen35MoEProfile(false);
    auto geometry = graphGeometry(DeviceId::rocm(0));
    geometry.resident_graph_rows = 9;
    geometry.apportioned_routed_experts = true;

    const auto add_expert_parent = [&](std::string suffix,
                                       std::size_t output_columns,
                                       std::size_t input_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = "blk.0." + std::move(suffix);
        tensor.elements = output_columns * input_columns *
                          static_cast<std::size_t>(profile.expert_count);
        tensor.K = input_columns;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add_expert_parent("ffn_gate_exps.weight", 512, 2048);
    add_expert_parent("ffn_up_exps.weight", 512, 2048);
    add_expert_parent("ffn_down_exps.weight", 2048, 512);

    std::size_t expected = 0u;
    for (const std::string_view format : {"F16", "BF16", "F32"})
    {
        for (TensorSizeInfo& tensor : profile.tensors)
            tensor.quant_type = format;
        const std::size_t actual =
            WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
                profile, geometry);
        if (expected == 0u)
            expected = actual;
        EXPECT_EQ(actual, expected) << "format=" << format;
    }

    WorkspaceRequirements base = MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/9,
        /*d_model=*/2048,
        /*intermediate=*/512,
        /*num_experts=*/256,
        /*top_k=*/8);
    base.merge(MoEWorkspaceBuffers::rocmMoE(
        /*max_seq_len=*/72,
        /*d_model=*/2048,
        /*intermediate=*/512,
        /*num_experts=*/256,
        /*top_k=*/1));
    constexpr std::size_t kPointerArrayCount = 3u;
    constexpr std::size_t kAlignedPointerArrayBytes = 256u;
    EXPECT_EQ(
        expected,
        base.total_bytes_with_alignment() +
            kPointerArrayCount * kAlignedPointerArrayBytes +
            floating_gemm_abi::kBlasMatmulWorkspaceBytes);
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

/**
 * @brief Largest-bucket admission must dominate intermediate attention peaks.
 *
 * CUDA's 4096-row direct plan needs no context summaries while its 3072-row
 * plan does. Use an otherwise small dense metadata profile so unrelated GEMM
 * storage cannot hide that missing contribution. Both vendors and TP through
 * eight participants owe the same graph-family capacity invariant.
 */
TEST(Test__WorkspaceMemoryEstimator, PrefillFamilyCoversIntermediateAttentionBuckets)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen2";
    profile.n_layers = 1;
    profile.d_model = 24 * 256;
    profile.d_ff = 1;
    profile.n_heads = 24;
    profile.n_kv_heads = 4;
    profile.head_dim = 256;
    profile.vocab_size = 1;
    profile.max_seq_len = 4096;
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        for (const int tp : {1, 2, 4, 8})
        {
            for (const int context : {4096, 16384})
            {
                auto geometry = graphGeometry(device, 1, tp);
                geometry.last_layer = 0;
                geometry.max_context_rows = context;
                const auto family = WorkspaceMemoryEstimator::estimate(profile, geometry);
                for (const int rows : {32, 128, 512, 1024, 2048, 3072})
                {
                    geometry.resident_graph_rows = rows;
                    EXPECT_GE(family, WorkspaceMemoryEstimator::estimate(profile, geometry))
                        << device.toString() << " TP=" << tp
                        << " context=" << context << " M=" << rows;
                }
            }
        }
    }
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
    installQwen122HybridLayerInventory(profile);

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
    installQwen122HybridLayerInventory(profile);

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
    geometry.local_query_head_start = 0;
    geometry.local_query_heads = 16;
    geometry.first_layer = 0;
    geometry.last_layer = 47;

    const size_t bytes = WorkspaceMemoryEstimator::estimate(
        profile, geometry);
    constexpr size_t kObservedExactDepth15SerialFamilyBytes =
        2288657668ULL;
    EXPECT_GE(bytes, kObservedExactDepth15SerialFamilyBytes)
        << "Depth-fifteen admission must price the retained main, grouped-verifier, and MTP namespaces before loading experts.";
}

/**
 * @brief Reproduce the retained ROCm TP4 matrix family that defeated preflight.
 *
 * The production Q8_K_XL model prepares every quantized codebook into the same
 * NativeVNNI ABI.  In particular, the mirrored vocabulary head needs a
 * 16-row split-K arena while the replicated MTP projector owns the widest K.
 * This inventory is metadata-only and therefore remains a fast device-free
 * regression for the exact late graph-materialization failure.
 */
TEST(Test__WorkspaceMemoryEstimator,
     Qwen122ROCmTP4Depth15CoversExactQuantizedGemmFamily)
{
    auto profile = qwen35MoEProfile(false);
    profile.architecture = "qwen35moe";
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.vocab_size = 248320;
    profile.expert_feed_forward_length = 1024;
    profile.mtp_layer_count = 1;
    installQwen122HybridLayerInventory(profile);

    const auto add_matrix = [&](std::string name,
                                int layer,
                                std::size_t n,
                                std::size_t k)
    {
        TensorSizeInfo tensor;
        tensor.name = std::move(name);
        tensor.quant_type = "Q8_K";
        tensor.elements = n * k;
        tensor.K = k;
        tensor.layer_index = layer;
        profile.tensors.push_back(std::move(tensor));
    };
    add_matrix("output.weight", -1, 248320, 3072);
    add_matrix("blk.48.nextn.eh_proj.weight", 48, 3072, 6144);
    add_matrix("blk.0.attn_qkv.weight", 0, 12288, 3072);
    add_matrix("blk.0.attn_gate.weight", 0, 8192, 3072);
    add_matrix("blk.0.ssm_alpha.weight", 0, 64, 3072);
    add_matrix("blk.0.ssm_beta.weight", 0, 64, 3072);
    add_matrix("blk.3.attn_q.weight", 3, 16384, 3072);
    add_matrix("blk.3.attn_k.weight", 3, 512, 3072);
    add_matrix("blk.3.attn_v.weight", 3, 512, 3072);

    /*
     * Routed parents retain all 256 experts in one 3-D inventory entry. Their
     * logical K is one expert matrix's K, while `elements` includes the outer
     * expert axis. This is the real GGUF geometry that exposed a 10+ GiB
     * phantom split-K charge when the expert count was mistaken for K.
     */
    const auto add_expert_parent = [&](std::string name,
                                       std::size_t n,
                                       std::size_t k)
    {
        TensorSizeInfo tensor;
        tensor.name = std::move(name);
        tensor.quant_type = "Q8_K";
        tensor.elements = n * k *
                          static_cast<std::size_t>(profile.expert_count);
        tensor.K = k;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add_expert_parent("blk.0.ffn_gate_exps.weight", 1024, 3072);
    add_expert_parent("blk.0.ffn_up_exps.weight", 1024, 3072);
    add_expert_parent("blk.0.ffn_down_exps.weight", 3072, 1024);

    auto geometry = graphGeometry(
        DeviceId::rocm(0), /*local_d_ff=*/256, /*total_shards=*/4);
    geometry.resident_graph_rows = 600;
    geometry.max_context_rows = 4096;
    geometry.shard_index = 0;
    geometry.has_exact_tensor_parallel_assignment = true;
    geometry.local_d_ff_start = 0;
    geometry.local_query_head_start = 0;
    geometry.local_query_heads = 8;
    geometry.local_kv_head_start = 0;
    geometry.local_kv_heads = 2;
    geometry.local_vocab_start = 0;
    geometry.local_vocab = 62080;
    geometry.first_layer = 0;
    geometry.last_layer = 47;
    geometry.mtp_target_query_rows = 16;
    geometry.mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;

    WorkspaceRequirements gemm_contract =
        rocm::quantized_gemm_workspace::projectionRequirements(
            /*rows=*/16, /*N=*/248320, /*K=*/3072);
    gemm_contract.merge(
        rocm::quantized_gemm_workspace::projectionRequirements(
            /*rows=*/600, /*N=*/3072, /*K=*/6144));
    const std::array<int, 4> gdn_fused_columns = {
        3072, 2048, 16, 16};
    rocm::quantized_gemm_workspace::appendFusedProjectionRequirements(
        gemm_contract, /*rows=*/600, gdn_fused_columns);

    EXPECT_EQ(
        gemm_contract.total_bytes_with_alignment(),
        1089370368ULL)
        << "The metadata contract must remain byte-identical to the runtime "
           "ROCm kernel ABI.";

    const std::size_t bytes =
        WorkspaceMemoryEstimator::estimate(profile, geometry);
    const size_t removed_unadmitted_summaries =
        rocmAttentionDescriptorBytes(geometry, 8, 2, 256, geometry.max_context_rows) -
        rocmAttentionDescriptorBytes(geometry, 8, 2, 256, geometry.resident_graph_rows);
    const std::size_t exact_runtime_serial_family_bytes =
        2546278148ULL - removed_unadmitted_summaries;
    EXPECT_GE(bytes, exact_runtime_serial_family_bytes)
        << "Preflight must cover the runtime interval plan before expert "
           "weights consume the remaining VRAM.";
    EXPECT_LE(
        bytes,
        exact_runtime_serial_family_bytes + 33ULL * 1024ULL * 1024ULL)
        << "The typed contract should not strand meaningful expert capacity "
           "behind a coarse safety reserve. The final MiB covers the routed "
           "fused-scatter ABI added by the explicit expert-parent inventory.";
}

/**
 * @brief Prove admission and graph stages share exact rollback bytes on both GPUs.
 *
 * Every rank of TP1/2/4/8 is included because GDN Q/K and value heads use a
 * linked modular assignment. A rank- or backend-specific approximation would
 * recreate the late allocation failure on a different topology.
 */
TEST(Test__WorkspaceMemoryEstimator,
     Qwen122Depth15GDNStateContractIsExactForCUDAAndROCmTP1ThroughTP8)
{
    auto profile = qwen35MoEProfile(true);
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.expert_feed_forward_length = 1024;
    installQwen122HybridLayerInventory(profile);

    constexpr std::size_t kGDNLayerCount = 36;
    for (const DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        for (const int tp : {1, 2, 4, 8})
        {
            for (int rank = 0; rank < tp; ++rank)
            {
                auto geometry = graphGeometry(
                    device,
                    /*local_d_ff=*/std::max(1, 1024 / tp),
                    tp);
                geometry.resident_graph_rows = 16;
                geometry.first_layer = 0;
                geometry.last_layer = 47;
                geometry.local_query_head_start = rank * (32 / tp);
                geometry.local_query_heads = 32 / tp;

                const std::size_t ordinary =
                    WorkspaceMemoryEstimator::estimate(profile, geometry);
                geometry.mtp_target_query_rows = 16;
                const std::size_t retained =
                    WorkspaceMemoryEstimator::estimate(profile, geometry);

                const auto assignment = GDNHeadAssignment::fromPartition(
                    /*global_key_heads=*/16,
                    /*global_value_heads=*/64,
                    geometry.local_query_head_start,
                    geometry.local_query_heads,
                    /*partition_total=*/32);
                const auto recurrence =
                    gdn_workspace::recurrenceStateFootprint(
                        /*slot_rows=*/16,
                        /*request_count=*/1,
                        assignment.localValueHeads(),
                        /*key_width=*/128,
                        /*value_width=*/128);
                const auto short_conv =
                    gdn_workspace::shortConvStateFootprint(
                        /*slot_rows=*/16,
                        /*request_count=*/1,
                        static_cast<int>(assignment.localFusedRows(128)),
                        /*kernel_size=*/4);
                const auto align256 = [](std::size_t bytes)
                {
                    return (bytes + 255u) & ~std::size_t{255u};
                };
                const std::size_t expected =
                    kGDNLayerCount *
                        (align256(recurrence.slot_bytes) +
                         align256(short_conv.slot_bytes)) +
                    align256(recurrence.work_bytes) +
                    align256(short_conv.work_bytes);

                EXPECT_EQ(retained - ordinary, expected)
                    << "device=" << device.toString()
                    << " tp=" << tp << " rank=" << rank;
            }
        }
    }
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
        kRedirectBytes + kThreeAlignedPointerArrays +
            floating_gemm_abi::kBlasMatmulWorkspaceBytes);
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
