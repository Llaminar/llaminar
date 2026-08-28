/**
 * @file WorkspaceMemoryEstimator.cpp
 * @brief Implements model-aware workspace reservation before GPU materialization.
 *
 * The estimator and the runtime allocator operate at different lifecycle
 * points but share the same kernel requirement formulas. This keeps GGUF
 * preflight aware of routed-expert, recurrent, and dense graph-family pressure
 * before weights consume the memory that exact graph capture will need.
 */

#include "planning/WorkspaceMemoryEstimator.h"

#include "config/GDNHeadAssignment.h"
#include "execution/compute_stages/stages/GDNSpeculativeWorkspaceContract.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include "kernels/cuda/attention/CUDAFlashAttentionLaunchPolicy.h"
#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"
#include "planning/WeightMemoryEstimator.h"

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{

namespace
{

/**
 * @brief Minimum physical envelope for the non-MoE production graph family.
 *
 * The exact routed-expert and recurrent contributions are added separately.
 * The remaining dense graph family includes terminal projection, attention,
 * quantization, decode K-partition, and library descriptor buffers. The 800
 * MiB floor is the measured upper envelope of those simultaneously published
 * names for current production graphs, rounded to a page-friendly boundary.
 */
constexpr size_t kMinimumGpuGraphFamilyBytes =
    800ULL * 1024ULL * 1024ULL;

size_t projectionOutputRows(
    const ModelMemoryProfile& profile,
    std::string_view name_fragment,
    int first_layer,
    int last_layer)
{
    size_t maximum = 0;
    for (const auto& tensor : profile.tensors)
    {
        if (tensor.layer_index < first_layer ||
            tensor.layer_index > last_layer ||
            tensor.name.find(name_fragment) == std::string::npos ||
            tensor.K == 0)
        {
            continue;
        }
        maximum = std::max(maximum, tensor.elements / tensor.K);
    }
    return maximum;
}

size_t shardColumns(size_t columns, int total_shards)
{
    const size_t shards = static_cast<size_t>(std::max(1, total_shards));
    return (columns + shards - 1) / shards;
}

bool hasHybridRecurrentLayer(
    const ModelMemoryProfile& profile,
    int first_layer,
    int last_layer)
{
    return std::any_of(
        profile.tensors.begin(),
        profile.tensors.end(),
        [&](const TensorSizeInfo& tensor)
        {
            return tensor.layer_index >= first_layer &&
                   tensor.layer_index <= last_layer &&
                   tensor.name.find(".ssm_out.weight") != std::string::npos;
        });
}

/**
 * @brief Count logical GDN layers whose rollback slots coexist in one verifier.
 *
 * One `.ssm_out.weight` marker is sufficient to identify a GDN layer and is
 * already part of the metadata-only tensor inventory.  A set protects the
 * admission result from aliases or additional prepared views of that tensor.
 */
std::size_t hybridRecurrentLayerCount(
    const ModelMemoryProfile& profile,
    int first_layer,
    int last_layer)
{
    std::set<int> layers;
    for (const auto& tensor : profile.tensors)
    {
        if (tensor.layer_index >= first_layer &&
            tensor.layer_index <= last_layer &&
            tensor.name.ends_with(".ssm_out.weight"))
        {
            layers.insert(tensor.layer_index);
        }
    }
    return layers.size();
}

/**
 * @brief Compute the exact MoE requirement contribution for one GPU participant.
 *
 * MoE scratch names coexist with the dense GEMM and recurrent names already
 * represented by the base estimate, so their aligned requirement is additive.
 * The same requirement factories are queried later by MoEExpertComputeStage;
 * using them here prevents metadata and graph construction from drifting.
 */
size_t exactMoEWorkspaceBytes(
    const ModelMemoryProfile& profile,
    int batch_size,
    int resident_graph_rows,
    DeviceId device)
{
    if (profile.expert_count == 0)
        return 0;

    if (profile.expert_count < 0 ||
        profile.expert_used_count <= 0 ||
        profile.expert_feed_forward_length <= 0)
    {
        throw std::runtime_error(
            "MoE workspace planning requires positive expert_count, "
            "expert_used_count, and expert_feed_forward_length");
    }

    const size_t batches = static_cast<size_t>(std::max(1, batch_size));
    const size_t rows =
        static_cast<size_t>(std::max(1, resident_graph_rows));
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) / batches)
    {
        throw std::runtime_error(
            "MoE workspace row envelope exceeds the supported integer geometry");
    }
    const int total_rows = static_cast<int>(rows * batches);

    WorkspaceRequirements requirements;
    if (device.is_cuda())
    {
        requirements = MoEWorkspaceBuffers::cudaMoE(
            total_rows,
            profile.d_model,
            profile.expert_feed_forward_length,
            profile.expert_count,
            profile.expert_used_count);
    }
    else if (device.is_rocm())
    {
        requirements = MoEWorkspaceBuffers::rocmMoE(
            total_rows,
            profile.d_model,
            profile.expert_feed_forward_length,
            profile.expert_count,
            profile.expert_used_count);
    }
    else
    {
        throw std::runtime_error(
            "MoE workspace planning received a non-GPU device after GPU sizing began");
    }
    return requirements.total_bytes_with_alignment();
}

/** @brief Add two byte counts while rejecting an unrepresentable plan. */
size_t checkedAdd(size_t left, size_t right, std::string_view contribution)
{
    if (right > std::numeric_limits<size_t>::max() - left)
    {
        throw std::runtime_error(
            "Workspace byte overflow while adding " +
            std::string(contribution));
    }
    return left + right;
}

/** @brief Multiply workspace cardinalities without permitting wraparound. */
size_t checkedMultiply(size_t left, size_t right, std::string_view contribution)
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    {
        throw std::runtime_error(
            "Workspace byte overflow while multiplying " +
            std::string(contribution));
    }
    return left * right;
}

/** @brief Apply the allocator's 256-byte persistent-buffer alignment. */
size_t alignedWorkspaceBytes(size_t bytes)
{
    constexpr size_t alignment = 256;
    if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1))
    {
        throw std::runtime_error(
            "Workspace byte overflow while applying buffer alignment");
    }
    return (bytes + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Price every persistent GDN rollback buffer in a grouped verifier.
 *
 * Transient main/verifier workspaces are event-ordered and can share the
 * serial-family allocation.  State snapshots cannot: each logical GDN layer
 * retains one independently addressable row bank until verification commits.
 * The stage-level byte arithmetic comes from GDNSpeculativeWorkspaceContract;
 * this function only applies TP geometry, layer multiplicity, and per-buffer
 * alignment.
 */
size_t exactHybridVerifierStateWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (geometry.mtp_target_query_rows <= 0)
        return 0;

    const std::size_t layer_count = hybridRecurrentLayerCount(
        profile, geometry.first_layer, geometry.last_layer);
    if (layer_count == 0u)
        return 0;
    if (profile.gdn_state_size <= 0 ||
        profile.gdn_group_count <= 0 ||
        profile.gdn_time_step_rank <= 0 ||
        profile.gdn_conv_kernel_size <= 1 ||
        profile.n_heads <= 0 || geometry.total_shards <= 0)
    {
        throw std::runtime_error(
            "Hybrid MTP workspace admission requires complete GDN metadata");
    }

    const int local_query_heads =
        geometry.local_query_heads > 0
            ? geometry.local_query_heads
            : profile.n_heads / geometry.total_shards;
    const int local_query_head_start =
        geometry.total_shards > 1
            ? geometry.local_query_head_start
            : 0;
    const auto assignment = GDNHeadAssignment::fromPartition(
        profile.gdn_group_count,
        profile.gdn_time_step_rank,
        local_query_head_start,
        local_query_heads,
        profile.n_heads);

    const std::size_t expected_inner = checkedMultiply(
        static_cast<std::size_t>(profile.gdn_time_step_rank),
        static_cast<std::size_t>(profile.gdn_state_size),
        "global GDN value geometry");
    if (profile.gdn_inner_size > 0 &&
        static_cast<std::size_t>(profile.gdn_inner_size) != expected_inner)
    {
        throw std::runtime_error(
            "Hybrid MTP workspace cannot represent GDN inner_size as value heads times state_size");
    }

    const std::size_t local_channels = assignment.localFusedRows(
        profile.gdn_state_size);
    if (local_channels >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error(
            "Participant-local GDN fused width exceeds int");
    }
    const auto recurrence = gdn_workspace::recurrenceStateFootprint(
        geometry.mtp_target_query_rows,
        std::max(1, geometry.batch_size),
        assignment.localValueHeads(),
        profile.gdn_state_size,
        profile.gdn_state_size);
    const auto short_conv = gdn_workspace::shortConvStateFootprint(
        geometry.mtp_target_query_rows,
        std::max(1, geometry.batch_size),
        static_cast<int>(local_channels),
        profile.gdn_conv_kernel_size);

    const std::size_t per_layer_slots = checkedAdd(
        alignedWorkspaceBytes(recurrence.slot_bytes),
        alignedWorkspaceBytes(short_conv.slot_bytes),
        "per-layer GDN verifier slots");
    size_t bytes = checkedMultiply(
        layer_count,
        per_layer_slots,
        "GDN layer count and verifier slots");
    bytes = checkedAdd(
        bytes,
        alignedWorkspaceBytes(recurrence.work_bytes),
        "shared GDN recurrence verifier work");
    bytes = checkedAdd(
        bytes,
        alignedWorkspaceBytes(short_conv.work_bytes),
        "shared GDN short-convolution verifier work");
    return bytes;
}

/** @brief Whether a source tensor selects the floating GEMM implementation. */
bool isFloatingWeight(const TensorSizeInfo& tensor)
{
    return tensor.quant_type == "F32" || tensor.quant_type == "F16" ||
           tensor.quant_type == "FP16" || tensor.quant_type == "BF16";
}

/** @brief Whether a GGUF parent tensor stacks every routed expert. */
bool isRoutedExpertWeight(std::string_view name)
{
    return name.ends_with(".ffn_gate_exps.weight") ||
           name.ends_with(".ffn_up_exps.weight") ||
           name.ends_with(".ffn_down_exps.weight");
}

/**
 * @brief Find the widest local floating projection in an assigned layer range.
 *
 * GGUF routed-expert parents add an expert axis outside K, so remove that axis
 * before interpreting output columns. Tensor-parallel output shards divide N;
 * reduction shards keep the complete N and only divide K. ExpertOverlay owner
 * slices retain whole experts and therefore never divide their projection N.
 */
size_t maximumFloatingProjectionColumns(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry,
    bool routed_only)
{
    size_t maximum = 0;
    for (const auto& tensor : profile.tensors)
    {
        const bool routed = isRoutedExpertWeight(tensor.name);
        if (tensor.layer_index < geometry.first_layer ||
            tensor.layer_index > geometry.last_layer ||
            (routed_only && !routed) || !isFloatingWeight(tensor) ||
            tensor.K == 0 || tensor.elements < tensor.K)
        {
            continue;
        }

        size_t columns = tensor.elements / tensor.K;
        if (routed)
        {
            if (profile.expert_count <= 0 ||
                columns % static_cast<size_t>(profile.expert_count) != 0)
            {
                throw std::runtime_error(
                    "Floating routed-expert workspace cannot recover one "
                    "expert projection width from " + tensor.name);
            }
            columns /= static_cast<size_t>(profile.expert_count);
        }

        const bool whole_overlay_expert =
            routed && geometry.apportioned_routed_experts;
        if (!whole_overlay_expert && geometry.total_shards > 1 &&
            WeightMemoryEstimator::tensorParallelShardAxis(tensor.name) ==
                TensorParallelWeightShardAxis::OutputColumns)
        {
            const size_t shards =
                static_cast<size_t>(geometry.total_shards);
            columns = (columns + shards - 1) / shards;
        }
        maximum = std::max(maximum, columns);
    }
    return maximum;
}

/**
 * @brief Exact CUDA floating-GEMM pointer and mapped-output workspace.
 *
 * The redirect is an eight-projection graph ABI, not a heuristic reserve. It
 * must use the same cardinality constant as CUDAFloatingPointGemmKernel.
 */
size_t cudaFloatingPointWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry,
    size_t execution_rows,
    bool routed_only)
{
    if (!geometry.device.is_cuda())
        return 0;

    const size_t columns = maximumFloatingProjectionColumns(
        profile, geometry, routed_only);
    if (columns == 0)
        return 0;

    const size_t projection_capacity =
        floating_gemm_abi::kMaxBatchedProjections;
    const size_t pointer_array_bytes = alignedWorkspaceBytes(
        checkedMultiply(
            projection_capacity,
            sizeof(float*),
            "CUDA floating GEMM pointer-array capacity"));
    size_t redirect_bytes = checkedMultiply(
        projection_capacity,
        execution_rows,
        "CUDA floating GEMM redirect projection rows");
    redirect_bytes = checkedMultiply(
        redirect_bytes,
        columns,
        "CUDA floating GEMM redirect columns");
    redirect_bytes = checkedMultiply(
        redirect_bytes,
        sizeof(float),
        "CUDA floating GEMM redirect element bytes");

    size_t total = alignedWorkspaceBytes(redirect_bytes);
    total = checkedAdd(total, pointer_array_bytes, "CUDA floating A pointers");
    total = checkedAdd(total, pointer_array_bytes, "CUDA floating B pointers");
    total = checkedAdd(total, pointer_array_bytes, "CUDA floating C pointers");
    return total;
}

/** @brief Whether this participant materializes at least one full-attention layer. */
bool hasFullAttentionLayer(
    const ModelMemoryProfile& profile,
    int first_layer,
    int last_layer)
{
    const bool explicit_full_attention = std::any_of(
        profile.tensors.begin(),
        profile.tensors.end(),
        [&](const TensorSizeInfo& tensor)
        {
            return tensor.layer_index >= first_layer &&
                   tensor.layer_index <= last_layer &&
                   tensor.name.ends_with(".attn_q.weight");
        });
    if (explicit_full_attention)
        return true;

    if (profile.full_attention_interval > 0)
    {
        for (int layer = std::max(0, first_layer);
             layer <= last_layer;
             ++layer)
        {
            if ((layer + 1) % profile.full_attention_interval == 0)
                return true;
        }
        return false;
    }

    /* A non-hybrid attention model uses full attention in every layer. */
    const bool model_is_hybrid = hasHybridRecurrentLayer(
        profile,
        0,
        std::max(0, profile.n_layers - 1));
    return !model_is_hybrid && profile.n_heads > 0 && profile.head_dim > 0;
}

/**
 * @brief Exact context-summary arena selected by backend capture policy.
 *
 * CUDA declares the current resident prefill graph. ROCm additionally declares
 * its non-monotonic geometry-selected family envelope, exactly as
 * AttentionComputeStage does before publishing the stable arena address.
 */
size_t exactAttentionWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (!geometry.device.is_gpu() ||
        !hasFullAttentionLayer(
            profile, geometry.first_layer, geometry.last_layer))
    {
        return 0;
    }
    const bool has_exact_local_query_heads =
        geometry.local_query_heads > 0;
    if (geometry.device_compute_units <= 0 || geometry.batch_size <= 0 ||
        geometry.resident_graph_rows <= 0 ||
        geometry.max_context_rows <= 0 || profile.n_heads <= 0 ||
        profile.head_dim <= 0 || geometry.total_shards <= 0 ||
        (!has_exact_local_query_heads &&
         profile.n_heads % geometry.total_shards != 0))
    {
        throw std::runtime_error(
            "Attention workspace planning requires positive graph/device "
            "geometry, physical SM/CU count, and either an exact local query "
            "head assignment or a uniformly divisible model head count");
    }

    const int local_query_heads =
        has_exact_local_query_heads
            ? geometry.local_query_heads
            : profile.n_heads / geometry.total_shards;
    size_t partial_output = 0;
    size_t partial_m = 0;
    size_t partial_l = 0;

    if (geometry.device.is_cuda())
    {
        const auto plan = cuda::fa2_policy::selectFA2PrefillParallelPlan({
            .batch_size = geometry.batch_size,
            .query_rows = geometry.resident_graph_rows,
            .local_query_heads = local_query_heads,
            .head_dim = profile.head_dim,
            .kv_capacity = geometry.max_context_rows,
            .sm_count = geometry.device_compute_units,
            .requested_axis =
                attention::AttentionPrefillParallelAxis::GeometrySelected,
        });
        if (!plan.valid)
        {
            throw std::runtime_error(
                "CUDA attention workspace policy rejected admitted graph geometry");
        }
        partial_output = plan.partial_output_bytes;
        partial_m = plan.partial_m_bytes;
        partial_l = plan.partial_l_bytes;
    }
    else if (geometry.device.is_rocm())
    {
        const rocm::fa2_policy::ROCmFA2PrefillParallelGeometry current{
            .batch_size = geometry.batch_size,
            .query_rows = geometry.resident_graph_rows,
            .local_query_heads = local_query_heads,
            .head_dim = profile.head_dim,
            .kv_capacity = geometry.max_context_rows,
            .compute_unit_count = geometry.device_compute_units,
            .lds_capacity_bytes =
                rocm::fa2_policy::kROCmFA2LDSCapacityBytes,
            .requested_axis =
                attention::AttentionPrefillParallelAxis::GeometrySelected,
        };
        const auto plan =
            rocm::fa2_policy::selectROCmFA2PrefillParallelPlan(current);
        if (!plan.valid)
        {
            throw std::runtime_error(
                "ROCm attention workspace policy rejected admitted graph geometry");
        }
        partial_output = plan.partial_output_bytes;
        partial_m = plan.partial_m_bytes;
        partial_l = plan.partial_l_bytes;

        auto family_geometry = current;
        family_geometry.query_rows = geometry.max_context_rows;
        const auto family = rocm::fa2_policy::
            selectROCmFA2GeometrySelectedWorkspaceEnvelope(family_geometry);
        if (family.valid && family.usesContextParallelism())
        {
            partial_output = std::max(
                partial_output, family.partial_output_bytes);
            partial_m = std::max(partial_m, family.partial_m_bytes);
            partial_l = std::max(partial_l, family.partial_l_bytes);
        }
    }

    size_t total = alignedWorkspaceBytes(partial_output);
    total = checkedAdd(
        total, alignedWorkspaceBytes(partial_m), "attention partial M");
    total = checkedAdd(
        total, alignedWorkspaceBytes(partial_l), "attention partial L");
    return total;
}

} // namespace

size_t WorkspaceMemoryEstimator::estimate(
    int batch_size,
    int max_seq_len,
    int d_model,
    int d_ff,
    int vocab_size,
    DeviceId device)
{
    // CPU kernels don't use a GPU workspace manager
    if (device.is_cpu())
    {
        return 0;
    }

    if (batch_size <= 0 || d_model <= 0 || vocab_size <= 0)
    {
        return 0;
    }
    size_t B = static_cast<size_t>(batch_size);
    size_t S = static_cast<size_t>(max_seq_len);
    size_t D = static_cast<size_t>(d_model);
    size_t F = static_cast<size_t>(d_ff);
    size_t V = static_cast<size_t>(vocab_size);

    constexpr size_t FP32 = 4;

    // Terminal-row LM head and padded-N library workspace.
    size_t lm_head_workspace = 3 * B * V * FP32;
    size_t padded_n = 8 * V * FP32;

    /*
     * The grouped-prefill NativeVNNI path owns four row-scaled workspaces for
     * the largest FFN projection: Q8 input (1 byte/value), one INT32
     * accumulator, three concurrent INT32 accumulator slots, and one FP32
     * temporary output. Together these are 21 bytes per output element.
     */
    size_t grouped_prefill = 21 * B * S * F;

    // Blockwise Q8 scales and sums: two FP32 values per 32-value K block.
    const size_t max_k = std::max(D, F);
    size_t blockwise_metadata =
        2 * FP32 * B * S * ((max_k + 31) / 32);

    /*
     * Decode K-partition partials, vendor-library descriptors, attention
     * conversion scratch, and other row-independent graph-family consumers.
     * The 320 MiB reserve is measured from the allocator BOM and remains
     * independent of prompt bucket rows.
     */
    constexpr size_t FIXED_GRAPH_FAMILY =
        320ULL * 1024ULL * 1024ULL;

    // Prepared embedding runs never reserve a vocab-by-hidden staging table.
    size_t raw =
        lm_head_workspace +
        padded_n +
        grouped_prefill +
        blockwise_metadata +
        FIXED_GRAPH_FAMILY;

    return std::max(kMinimumGpuGraphFamilyBytes, raw);
}

size_t WorkspaceMemoryEstimator::estimate(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    size_t bytes = estimate(
        geometry.batch_size,
        geometry.resident_graph_rows,
        profile.d_model,
        geometry.local_d_ff,
        profile.vocab_size,
        geometry.device);
    if (bytes == 0)
        return 0;

    bytes = checkedAdd(
        bytes,
        exactMoEWorkspaceBytes(
            profile,
            geometry.batch_size,
            geometry.resident_graph_rows,
            geometry.device),
        "MoE graph-family requirements");

    bytes = checkedAdd(
        bytes,
        exactAttentionWorkspaceBytes(profile, geometry),
        "attention context-summary requirements");

    const size_t execution_rows = checkedMultiply(
        static_cast<size_t>(std::max(1, geometry.batch_size)),
        static_cast<size_t>(std::max(1, geometry.resident_graph_rows)),
        "floating projection execution rows");
    bytes = checkedAdd(
        bytes,
        cudaFloatingPointWorkspaceBytes(
            profile, geometry, execution_rows, /*routed_only=*/false),
        "CUDA floating projection requirements");

    if (hasHybridRecurrentLayer(
            profile, geometry.first_layer, geometry.last_layer))
    {
        const size_t gdn_qkv = shardColumns(
            projectionOutputRows(
                profile,
                ".attn_qkv.weight",
                geometry.first_layer,
                geometry.last_layer),
            geometry.total_shards);
        const size_t gdn_gate = shardColumns(
            projectionOutputRows(
                profile,
                ".attn_gate.weight",
                geometry.first_layer,
                geometry.last_layer),
            geometry.total_shards);
        if (gdn_qkv != 0 || gdn_gate != 0)
        {
            /*
             * Hybrid GDN layers add one in-place short-convolution scratch
             * row of QKV width and one three-way deinterleave scratch row of
             * gate width.
             */
            const size_t hybrid_row_bytes =
                (gdn_qkv + 3 * gdn_gate) * sizeof(float);
            const size_t hybrid_bytes =
                checkedMultiply(execution_rows, hybrid_row_bytes,
                                "hybrid recurrent row scratch");
            bytes = checkedAdd(
                bytes, hybrid_bytes, "hybrid recurrent scratch");
        }
    }

    if (geometry.mtp_target_query_rows > 0)
    {
        /*
         * The exact runtime interval planner can alias many scratch names
         * across the main and compact MTP participants, but it must retain
         * different namespaces and initialize-once publications together.
         * Before graph construction, two independently complete envelopes are
         * therefore the safe compositional bound. Clear the MTP dimension on
         * the recursive call so the compact envelope is added exactly once.
         */
        WorkspaceMemoryGeometry mtp_geometry = geometry;
        mtp_geometry.resident_graph_rows =
            std::max(1, geometry.mtp_target_query_rows);
        mtp_geometry.mtp_target_query_rows = 0;
        const size_t compact_family_bytes = estimate(profile, mtp_geometry);
        const size_t persistent_hybrid_snapshot_bytes =
            exactHybridVerifierStateWorkspaceBytes(profile, geometry);
        if (persistent_hybrid_snapshot_bytes > 0)
        {
            /*
             * Main and compact graph participants are serial and therefore
             * share the larger transient envelope. Per-layer rollback slots
             * remain live across that participant and are added exactly once.
             */
            bytes = std::max(bytes, compact_family_bytes);
            bytes = checkedAdd(
                bytes,
                persistent_hybrid_snapshot_bytes,
                "retained hybrid MTP rollback state");
        }
        else
        {
            /*
             * Non-hybrid families retain the historical conservative sum;
             * they have no typed per-layer state contract that proves a more
             * aggressive alias relationship.
             */
            bytes = checkedAdd(
                bytes,
                compact_family_bytes,
                "retained compact MTP graph family");
        }
    }

    return bytes;
}

size_t WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (geometry.device.is_cpu())
        return 0;
    if (!geometry.device.is_gpu())
    {
        throw std::runtime_error(
            "Routed-expert participant workspace requires a valid CPU or GPU device");
    }
    if (profile.expert_count <= 0 ||
        profile.expert_used_count <= 0 ||
        profile.expert_feed_forward_length <= 0 ||
        profile.d_model <= 0)
    {
        throw std::runtime_error(
            "Routed-expert participant workspace requires complete MoE geometry");
    }

    const size_t batches =
        static_cast<size_t>(std::max(1, geometry.batch_size));
    const size_t rows =
        static_cast<size_t>(std::max(1, geometry.resident_graph_rows));
    const size_t top_k = static_cast<size_t>(profile.expert_used_count);
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) /
                   batches ||
        rows * batches >
            static_cast<size_t>(std::numeric_limits<int>::max()) / top_k)
    {
        throw std::runtime_error(
            "Routed-expert participant compact-row envelope exceeds integer geometry");
    }

    const int direct_rows = static_cast<int>(rows * batches);
    const int compact_rows = static_cast<int>(rows * batches * top_k);

    /*
     * A node-local mapped follower retains MoEExpertComputeStage directly and
     * therefore declares `(direct_rows, model_top_k)`. CPU and other explicit
     * heterogeneous boundaries use MoELocalExpertStage, which declares one
     * compact row per route as `(compact_rows, 1)`. The ROCm requirement is not
     * algebraically interchangeable: router scratch scales with token rows,
     * while grouped execution scratch also depends on top-k. Merge by stable
     * workspace name exactly as the runtime serial-family allocator does.
     */
    WorkspaceRequirements direct_requirements;
    WorkspaceRequirements compact_requirements;
    if (geometry.device.is_cuda())
    {
        direct_requirements = MoEWorkspaceBuffers::cudaMoE(
            direct_rows,
            profile.d_model,
            profile.expert_feed_forward_length,
            profile.expert_count,
            profile.expert_used_count);
        compact_requirements = MoEWorkspaceBuffers::cudaMoE(
            compact_rows,
            profile.d_model,
            profile.expert_feed_forward_length,
            profile.expert_count,
            /*top_k=*/1);
    }
    else
    {
        direct_requirements = MoEWorkspaceBuffers::rocmMoE(
            direct_rows,
            profile.d_model,
            profile.expert_feed_forward_length,
            profile.expert_count,
            profile.expert_used_count);
        compact_requirements = MoEWorkspaceBuffers::rocmMoE(
            compact_rows,
            profile.d_model,
            profile.expert_feed_forward_length,
            profile.expert_count,
            /*top_k=*/1);
    }
    direct_requirements.merge(compact_requirements);
    size_t bytes = direct_requirements.total_bytes_with_alignment();
    WorkspaceMemoryGeometry expert_geometry = geometry;
    expert_geometry.apportioned_routed_experts = true;
    bytes = checkedAdd(
        bytes,
        cudaFloatingPointWorkspaceBytes(
            profile,
            expert_geometry,
            static_cast<size_t>(compact_rows),
            /*routed_only=*/true),
        "CUDA local floating-expert requirements");
    return bytes;
}

} // namespace llaminar2
