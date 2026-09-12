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
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include "kernels/common/EmbeddingWorkspaceContract.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "kernels/cuda/attention/CUDAFlashAttentionWorkspaceEnvelope.h"
#include "kernels/kvcache/KVCacheWorkspaceContract.h"
#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmWorkspaceContract.h"
#include "kernels/rope/RoPEWorkspaceContract.h"
#include "utils/VramBillOfMaterials.h"

#include <algorithm>
#include <array>
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
    /*
     * GPU recurrence kernels publish graph-stable staging rows beside the
     * rollback slots. CPU kernels update their slot banks directly and do not
     * declare those work buffers. Keep the backend distinction in this shared
     * metadata contract so preflight does not invent CPU-only padding.
     */
    if (geometry.device.is_gpu())
    {
        bytes = checkedAdd(
            bytes,
            alignedWorkspaceBytes(recurrence.work_bytes),
            "shared GDN recurrence verifier work");
        bytes = checkedAdd(
            bytes,
            alignedWorkspaceBytes(short_conv.work_bytes),
            "shared GDN short-convolution verifier work");
    }
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

/** @brief Whether a tensor is prepared through the quantized NativeVNNI GEMM. */
bool isQuantizedMatrixWeight(const TensorSizeInfo& tensor)
{
    return tensor.K > 0u && tensor.elements >= tensor.K &&
           tensor.elements % tensor.K == 0u &&
           !tensor.quant_type.empty() && !isFloatingWeight(tensor);
}

/** @brief Whether a matrix supplies the terminal vocabulary projection. */
bool isTerminalProjectionWeight(std::string_view name)
{
    return name == "output.weight" ||
           name.find("lm_head") != std::string_view::npos;
}

/** @brief Whether a matrix is an embedding lookup rather than a GEMM weight. */
bool isEmbeddingWeight(std::string_view name)
{
    return name.find("token_embd") != std::string_view::npos ||
           name.find("embed_tokens") != std::string_view::npos;
}

/** @brief One exact interval on a schema-owned logical TP dimension. */
struct WorkspaceLogicalInterval
{
    std::size_t start = 0u;
    std::size_t count = 0u;
    std::size_t total = 0u;
};

/** @brief One participant-local GEMM matrix shape. */
struct WorkspaceMatrixShape
{
    int output_columns = 0;
    int input_columns = 0;

    [[nodiscard]] bool valid() const noexcept
    {
        return output_columns > 0 && input_columns > 0;
    }
};

/** @brief Narrow a checked workspace cardinality to the kernel's integer ABI. */
int checkedWorkspaceDimension(std::size_t value, std::string_view name)
{
    if (value == 0u ||
        value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error(
            "Workspace matrix " + std::string(name) +
            " is outside the positive int range");
    }
    return static_cast<int>(value);
}

/** @brief Divide one semantic TP dimension with remainder ownership. */
WorkspaceLogicalInterval equalWorkspaceInterval(
    int total,
    const WorkspaceMemoryGeometry& geometry,
    std::string_view name)
{
    if (total <= 0 || geometry.total_shards <= 0 ||
        geometry.shard_index < 0 ||
        geometry.shard_index >= geometry.total_shards)
    {
        throw std::runtime_error(
            "Workspace cannot divide invalid " + std::string(name) +
            " TP geometry");
    }
    const std::size_t extent = static_cast<std::size_t>(total);
    const std::size_t degree =
        static_cast<std::size_t>(geometry.total_shards);
    const std::size_t index =
        static_cast<std::size_t>(geometry.shard_index);
    const std::size_t quotient = extent / degree;
    const std::size_t remainder = extent % degree;
    const std::size_t count = quotient + (index < remainder ? 1u : 0u);
    if (count == 0u)
    {
        throw std::runtime_error(
            "Workspace TP degree exceeds " + std::string(name));
    }
    return {
        .start = index * quotient + std::min(index, remainder),
        .count = count,
        .total = extent,
    };
}

/**
 * @brief Resolve the exact local interval for a schema dimension.
 *
 * Explicit rank-planner assignments win. Uniform TP uses the same quotient
 * and remainder rule as weight preparation; it never guesses by flat bytes.
 */
WorkspaceLogicalInterval workspaceLogicalInterval(
    WeightDimensionType dimension,
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    int total = 0;
    int start = 0;
    int count = 0;
    switch (dimension)
    {
    case WeightDimensionType::Heads:
    case WeightDimensionType::ProportionalHeads:
    case WeightDimensionType::FusedQKVHeads:
        total = profile.n_heads;
        start = geometry.local_query_head_start;
        count = geometry.local_query_heads;
        break;
    case WeightDimensionType::KVHeads:
        total = profile.n_kv_heads;
        start = geometry.local_kv_head_start;
        count = geometry.local_kv_heads;
        break;
    case WeightDimensionType::FFNHidden:
        total = profile.d_ff;
        start = geometry.local_d_ff_start;
        count = geometry.local_d_ff;
        break;
    case WeightDimensionType::Vocab:
        total = profile.vocab_size;
        start = geometry.local_vocab_start;
        count = geometry.local_vocab;
        break;
    case WeightDimensionType::Bias1D:
    case WeightDimensionType::None:
        throw std::runtime_error(
            "Sharded workspace matrix lacks a schema-owned dimension");
    }

    if (!geometry.has_exact_tensor_parallel_assignment)
    {
        return equalWorkspaceInterval(total, geometry, "logical dimension");
    }
    if (total <= 0 || start < 0 || count <= 0 || start > total - count)
    {
        throw std::runtime_error(
            "Workspace received an invalid exact TP assignment");
    }
    return {
        .start = static_cast<std::size_t>(start),
        .count = static_cast<std::size_t>(count),
        .total = static_cast<std::size_t>(total),
    };
}

/** @brief Apply a semantic interval to one physical matrix axis. */
std::size_t workspaceAxisExtent(
    std::size_t complete,
    WorkspaceLogicalInterval interval,
    std::string_view name)
{
    if (complete == 0u || interval.total == 0u ||
        interval.start > interval.total - interval.count)
    {
        throw std::runtime_error(
            "Workspace cannot slice invalid " + std::string(name));
    }
    const std::size_t begin = checkedMultiply(
        complete, interval.start, name) / interval.total;
    const std::size_t end = checkedMultiply(
        complete, interval.start + interval.count, name) / interval.total;
    if (end <= begin)
    {
        throw std::runtime_error(
            "Workspace TP slice is empty for " + std::string(name));
    }
    return end - begin;
}

/** @brief Whether a layer belongs to the retained replicated MTP sidecar. */
bool isRetainedMTPSidecarLayer(
    const TensorSizeInfo& tensor,
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    return geometry.mtp_target_query_rows > 0 &&
           profile.mtp_layer_count > 0 && tensor.layer_index >= 0 &&
           tensor.layer_index >= profile.n_layers - profile.mtp_layer_count;
}

/** @brief Whether the continuation graph materializes this matrix. */
bool workspaceOwnsTensor(
    const TensorSizeInfo& tensor,
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    return tensor.layer_index < 0 ||
           (tensor.layer_index >= geometry.first_layer &&
            tensor.layer_index <= geometry.last_layer) ||
           isRetainedMTPSidecarLayer(tensor, profile, geometry);
}

/**
 * @brief Resolve the exact local N/K shape used by prepared weight loading.
 *
 * The MTP sidecar's dense block is deliberately replicated per participant.
 * Routed-expert parents are divided by expert identity rather than TP matrix
 * axes, because one prepared expert kernel always sees one complete matrix.
 */
WorkspaceMatrixShape localWorkspaceMatrixShape(
    const TensorSizeInfo& tensor,
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry,
    const WeightShardingConfig& sharding,
    bool force_full_terminal)
{
    if (tensor.K == 0u || tensor.elements < tensor.K ||
        tensor.elements % tensor.K != 0u)
    {
        return {};
    }
    std::size_t output_columns = tensor.elements / tensor.K;
    std::size_t input_columns = tensor.K;

    if (isRoutedExpertWeight(tensor.name))
    {
        if (profile.expert_count <= 0 ||
            output_columns % static_cast<std::size_t>(profile.expert_count) !=
                0u)
        {
            throw std::runtime_error(
                "Routed expert workspace cannot recover one matrix from " +
                tensor.name);
        }
        output_columns /= static_cast<std::size_t>(profile.expert_count);
        return {
            .output_columns = checkedWorkspaceDimension(
                output_columns, "expert output columns"),
            .input_columns = checkedWorkspaceDimension(
                input_columns, "expert input columns"),
        };
    }

    const bool replicated_sidecar =
        isRetainedMTPSidecarLayer(tensor, profile, geometry);
    if (geometry.total_shards > 1 && !replicated_sidecar &&
        !(force_full_terminal && isTerminalProjectionWeight(tensor.name)))
    {
        const auto [mode, dimension] =
            sharding.getModeAndDimension(tensor.name);
        if (mode != WeightShardingMode::Replicate)
        {
            if (dimension == WeightDimensionType::FusedQKVHeads &&
                (mode == WeightShardingMode::ColumnParallel ||
                 mode == WeightShardingMode::RowParallel))
            {
                const std::size_t gdn_rows =
                    profile.gdn_group_count > 0 &&
                            profile.gdn_time_step_rank > 0 &&
                            profile.gdn_state_size > 0
                        ? static_cast<std::size_t>(
                              2 * profile.gdn_group_count +
                              profile.gdn_time_step_rank) *
                              static_cast<std::size_t>(profile.gdn_state_size)
                        : 0u;
                const std::size_t attention_rows =
                    profile.n_heads > 0 && profile.n_kv_heads > 0 &&
                            profile.head_dim > 0
                        ? static_cast<std::size_t>(
                              profile.n_heads + 2 * profile.n_kv_heads) *
                              static_cast<std::size_t>(profile.head_dim)
                        : 0u;
                if (output_columns == gdn_rows)
                {
                    const auto assignment = GDNHeadAssignment::fromPartition(
                        profile.gdn_group_count,
                        profile.gdn_time_step_rank,
                        geometry.local_query_head_start,
                        geometry.local_query_heads,
                        profile.n_heads);
                    output_columns = assignment.localFusedRows(
                        profile.gdn_state_size);
                }
                else if (output_columns == attention_rows)
                {
                    output_columns =
                        static_cast<std::size_t>(
                            geometry.local_query_heads +
                            2 * geometry.local_kv_heads) *
                        static_cast<std::size_t>(profile.head_dim);
                }
                else
                {
                    output_columns = workspaceAxisExtent(
                        output_columns,
                        workspaceLogicalInterval(
                            dimension, profile, geometry),
                        "fused-QKV output axis");
                }
            }
            else if (mode == WeightShardingMode::InputParallel)
            {
                input_columns = workspaceAxisExtent(
                    input_columns,
                    workspaceLogicalInterval(
                        dimension, profile, geometry),
                    "input-parallel K axis");
            }
            else if (mode == WeightShardingMode::ColumnParallel ||
                     mode == WeightShardingMode::RowParallel)
            {
                output_columns = workspaceAxisExtent(
                    output_columns,
                    workspaceLogicalInterval(
                        dimension, profile, geometry),
                    "output-parallel N axis");
            }
            else if (mode != WeightShardingMode::ExpertIdApportioned)
            {
                throw std::runtime_error(
                    "Workspace encountered an unsupported weight-sharding mode");
            }
        }
    }

    return {
        .output_columns = checkedWorkspaceDimension(
            output_columns, "output columns"),
        .input_columns = checkedWorkspaceDimension(
            input_columns, "input columns"),
    };
}

/** @brief Find one exact layer-local matrix suffix in the metadata inventory. */
const TensorSizeInfo* findLayerWorkspaceTensor(
    const ModelMemoryProfile& profile,
    int layer,
    std::string_view suffix)
{
    const auto found = std::find_if(
        profile.tensors.begin(),
        profile.tensors.end(),
        [&](const TensorSizeInfo& candidate)
        {
            return candidate.layer_index == layer &&
                   candidate.name.ends_with(suffix);
        });
    return found == profile.tensors.end() ? nullptr : &*found;
}

/**
 * @brief Price ROCm's exact codebook-independent quantized GEMM name set.
 *
 * Each source matrix contributes its real participant-local N/K geometry.
 * Requirements merge by stable name exactly as graph construction does. Fused
 * QKV, GDN, gate/up, and routed-expert bundles additionally declare the
 * simultaneous scatter bank used by their side-stream transaction.
 */
size_t exactROCmQuantizedGemmWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (!geometry.device.is_rocm())
        return 0u;

    const WeightShardingConfig sharding =
        SchemaFactoryRegistry::getWeightShardingConfig(profile.architecture);
    const int prefill_rows = checkedWorkspaceDimension(
        checkedMultiply(
            static_cast<std::size_t>(std::max(1, geometry.batch_size)),
            static_cast<std::size_t>(std::max(1, geometry.resident_graph_rows)),
            "ROCm prefill rows"),
        "ROCm prefill rows");
    const int terminal_rows = std::max(
        1,
        geometry.mtp_target_query_rows > 0
            ? geometry.mtp_target_query_rows
            : geometry.batch_size);
    const bool full_terminal =
        geometry.mtp_terminal_logits_layout ==
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;

    WorkspaceRequirements requirements;
    bool has_explicit_terminal = false;
    const TensorSizeInfo* tied_embedding = nullptr;
    for (const auto& tensor : profile.tensors)
    {
        if (!workspaceOwnsTensor(tensor, profile, geometry))
            continue;
        if (isEmbeddingWeight(tensor.name))
        {
            if (isQuantizedMatrixWeight(tensor))
                tied_embedding = &tensor;
            continue;
        }
        if (!isQuantizedMatrixWeight(tensor))
            continue;

        const bool terminal = isTerminalProjectionWeight(tensor.name);
        has_explicit_terminal = has_explicit_terminal || terminal;
        const bool mtp_sidecar = isRetainedMTPSidecarLayer(
            tensor, profile, geometry);
        const int projection_rows =
            terminal || mtp_sidecar ? terminal_rows : prefill_rows;
        const WorkspaceMatrixShape shape = localWorkspaceMatrixShape(
            tensor, profile, geometry, sharding, full_terminal);
        if (!shape.valid())
            continue;
        logVramBomLine(
            "workspace_admission_rocm_projection",
            "device=" + geometry.device.toString() +
                " tensor=" + tensor.name +
                " rows=" +
                std::to_string(projection_rows) +
                " output_columns=" +
                std::to_string(shape.output_columns) +
                " input_columns=" +
                std::to_string(shape.input_columns) +
                " terminal=" + (terminal ? "true" : "false"));
        requirements.merge(
            rocm::quantized_gemm_workspace::projectionRequirements(
                projection_rows,
                shape.output_columns,
                shape.input_columns));
    }

    if (!has_explicit_terminal && tied_embedding)
    {
        TensorSizeInfo terminal = *tied_embedding;
        terminal.name = "output.weight";
        const WorkspaceMatrixShape shape = localWorkspaceMatrixShape(
            terminal, profile, geometry, sharding, full_terminal);
        requirements.merge(
            rocm::quantized_gemm_workspace::projectionRequirements(
                terminal_rows,
                shape.output_columns,
                shape.input_columns));
    }

    for (const auto& anchor : profile.tensors)
    {
        if (!workspaceOwnsTensor(anchor, profile, geometry) ||
            !isQuantizedMatrixWeight(anchor))
        {
            continue;
        }

        std::vector<int> fused_columns;
        if (anchor.name.ends_with(".attn_qkv.weight"))
        {
            constexpr std::array<std::string_view, 4> suffixes = {
                ".attn_qkv.weight",
                ".attn_gate.weight",
                ".ssm_alpha.weight",
                ".ssm_beta.weight"};
            for (const std::string_view suffix : suffixes)
            {
                const TensorSizeInfo* member = findLayerWorkspaceTensor(
                    profile, anchor.layer_index, suffix);
                if (!member)
                {
                    fused_columns.clear();
                    break;
                }
                const WorkspaceMatrixShape shape = localWorkspaceMatrixShape(
                    *member, profile, geometry, sharding, false);
                if (!shape.valid())
                {
                    fused_columns.clear();
                    break;
                }
                fused_columns.push_back(shape.output_columns);
            }
        }
        else if (anchor.name.ends_with(".attn_q.weight"))
        {
            constexpr std::array<std::string_view, 3> suffixes = {
                ".attn_q.weight",
                ".attn_k.weight",
                ".attn_v.weight"};
            for (const std::string_view suffix : suffixes)
            {
                const TensorSizeInfo* member = findLayerWorkspaceTensor(
                    profile, anchor.layer_index, suffix);
                if (!member)
                {
                    fused_columns.clear();
                    break;
                }
                const WorkspaceMatrixShape shape = localWorkspaceMatrixShape(
                    *member, profile, geometry, sharding, false);
                if (!shape.valid())
                {
                    fused_columns.clear();
                    break;
                }
                fused_columns.push_back(shape.output_columns);
            }
        }
        else if (anchor.name.ends_with(".ffn_gate_exps.weight"))
        {
            const WorkspaceMatrixShape shape = localWorkspaceMatrixShape(
                anchor, profile, geometry, sharding, false);
            fused_columns.assign(
                static_cast<std::size_t>(
                    std::max(1, profile.expert_used_count)) * 2u,
                shape.output_columns);
        }
        else if (anchor.name.find(".ffn_gate") != std::string::npos &&
                 anchor.name.ends_with(".weight"))
        {
            std::string up_name = anchor.name;
            const std::size_t gate = up_name.find(".ffn_gate");
            up_name.replace(gate, std::string_view(".ffn_gate").size(),
                            ".ffn_up");
            const auto up = std::find_if(
                profile.tensors.begin(),
                profile.tensors.end(),
                [&](const TensorSizeInfo& candidate)
                { return candidate.name == up_name; });
            if (up != profile.tensors.end())
            {
                const WorkspaceMatrixShape gate_shape =
                    localWorkspaceMatrixShape(
                        anchor, profile, geometry, sharding, false);
                const WorkspaceMatrixShape up_shape =
                    localWorkspaceMatrixShape(
                        *up, profile, geometry, sharding, false);
                fused_columns = {
                    gate_shape.output_columns,
                    up_shape.output_columns};
            }
        }

        if (!fused_columns.empty())
        {
            WorkspaceRequirements fused_requirements;
            const int fused_rows = isRetainedMTPSidecarLayer(
                                       anchor, profile, geometry)
                                       ? terminal_rows
                                       : prefill_rows;
            rocm::quantized_gemm_workspace::appendFusedProjectionRequirements(
                fused_requirements, fused_rows, fused_columns);
            /*
             * Every layer binds the same stable fused-scatter ABI name. The
             * runtime workspace allocator canonicalizes that name across
             * graph consumers and publishes only the widest extent. Preserve
             * that ownership rule during metadata admission: appending
             * directly to the aggregate would charge one physical buffer per
             * layer even though no such allocations exist at materialization.
             */
            requirements.merge(fused_requirements);
        }
    }

    return requirements.total_bytes_with_alignment();
}

/**
 * @brief Compose the exact ROCm GEMM ABI for both routed endpoint graph forms.
 *
 * A mapped GPU follower retains `MoEExpertComputeStage`: its gate projection
 * owns the ordinary NativeVNNI buffers and anchors the simultaneous gate/up
 * side-stream arena.  A compact boundary follower retains
 * `MoELocalExpertStage`, which queries gate, up, and down projections
 * independently after flattening one row per route.  Production may retain
 * either endpoint form for the same admitted device, so this metadata-only
 * composition mirrors both stage declarations by stable workspace name.
 *
 * Every quantized GGUF codebook reaches the same prepared NativeVNNI ABI.
 * Floating ROCm projections need only the three bounded device pointer arrays;
 * their scalar type does not change that graph-captured address contract.
 */
WorkspaceRequirements rocmRoutedExpertGemmRequirements(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry,
    int direct_rows,
    int compact_rows)
{
    WorkspaceRequirements direct;
    WorkspaceRequirements compact;
    bool quantized_gate_anchor = false;
    bool floating_gate_anchor = false;
    bool floating_compact_projection = false;

    const WeightShardingConfig sharding =
        SchemaFactoryRegistry::getWeightShardingConfig(profile.architecture);
    for (const TensorSizeInfo& tensor : profile.tensors)
    {
        if (!workspaceOwnsTensor(tensor, profile, geometry) ||
            !isRoutedExpertWeight(tensor.name))
        {
            continue;
        }

        const bool gate = tensor.name.ends_with(".ffn_gate_exps.weight");
        if (isQuantizedMatrixWeight(tensor))
        {
            const WorkspaceMatrixShape shape = localWorkspaceMatrixShape(
                tensor, profile, geometry, sharding,
                /*force_full_terminal=*/false);
            if (!shape.valid())
            {
                throw std::runtime_error(
                    "ROCm routed-expert workspace cannot recover a prepared matrix shape from " +
                    tensor.name);
            }

            compact.merge(
                rocm::quantized_gemm_workspace::projectionRequirements(
                    compact_rows,
                    shape.output_columns,
                    shape.input_columns));
            if (gate)
            {
                quantized_gate_anchor = true;
                direct.merge(
                    rocm::quantized_gemm_workspace::projectionRequirements(
                        direct_rows,
                        shape.output_columns,
                        shape.input_columns));
            }
        }
        else if (isFloatingWeight(tensor))
        {
            floating_compact_projection = true;
            floating_gate_anchor = floating_gate_anchor || gate;
        }
    }

    if (quantized_gate_anchor)
    {
        const std::vector<int> fused_columns(
            static_cast<std::size_t>(profile.expert_used_count) * 2u,
            profile.expert_feed_forward_length);
        rocm::quantized_gemm_workspace::appendFusedProjectionRequirements(
            direct, direct_rows, fused_columns);
    }

    const auto append_floating_pointer_abi = [](WorkspaceRequirements& target)
    {
        target.buffers.push_back({
            floating_gemm_abi::kROCmBlasMatmulWorkspace,
            floating_gemm_abi::kBlasMatmulWorkspaceBytes, 256, true});
        const std::size_t bytes =
            floating_gemm_abi::kMaxBatchedProjections * sizeof(float*);
        target.buffers.push_back({
            GemmWorkspaceBuffers::ROCM_FP32_BATCH_A_PTRS,
            bytes, 256, true});
        target.buffers.push_back({
            GemmWorkspaceBuffers::ROCM_FP32_BATCH_B_PTRS,
            bytes, 256, true});
        target.buffers.push_back({
            GemmWorkspaceBuffers::ROCM_FP32_BATCH_C_PTRS,
            bytes, 256, true});
    };
    if (floating_gate_anchor)
        append_floating_pointer_abi(direct);
    if (floating_compact_projection)
        append_floating_pointer_abi(compact);

    direct.merge(compact);
    return direct;
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
    std::optional<WeightShardingConfig> sharding;
    if (geometry.total_shards > 1)
    {
        sharding = SchemaFactoryRegistry::getWeightShardingConfig(
            profile.architecture);
    }
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
        if (!whole_overlay_expert && sharding.has_value())
        {
            const WeightShardingMode mode = sharding->getMode(tensor.name);
            if (mode == WeightShardingMode::ColumnParallel ||
                mode == WeightShardingMode::RowParallel)
            {
                const size_t shards =
                    static_cast<size_t>(geometry.total_shards);
                columns = (columns + shards - 1) / shards;
            }
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
size_t gpuFloatingPointWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry,
    size_t execution_rows,
    bool routed_only)
{
    if (!geometry.device.is_gpu())
        return 0;

    const size_t columns = maximumFloatingProjectionColumns(
        profile, geometry, routed_only);
    if (columns == 0)
        return 0;

    // The routed-only ROCm caller already composes the descriptor-based
    // direct/compact GEMM contract. Dense/continuation graphs need its same
    // BLAS region and pointer ABI here, without a second live ledger.
    if (geometry.device.is_rocm())
        return routed_only ? 0u : floating_gemm_abi::kBlasMatmulWorkspaceBytes +
            3u * alignedWorkspaceBytes(
                floating_gemm_abi::kMaxBatchedProjections * sizeof(float *));

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
    total = checkedAdd(total, floating_gemm_abi::kBlasMatmulWorkspaceBytes,
                       "CUDA context-borrowing BLAS stage workspace");
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
 * @brief Exact complete attention arena selected by backend capture policy.
 *
 * CUDA declares the current resident prefill graph. ROCm additionally declares
 * its non-monotonic geometry-selected family envelope, exactly as
 * AttentionComputeStage does before publishing the stable arena address. The
 * canonical contract also retains split-decode, device-param, and FP32 K/V
 * conversion buffers so admission and the runtime kernel cannot diverge.
 */
size_t exactAttentionWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (!hasFullAttentionLayer(
            profile, geometry.first_layer, geometry.last_layer))
    {
        return 0;
    }
    const bool has_exact_local_query_heads =
        geometry.local_query_heads > 0;
    if (geometry.device_compute_units <= 0 || geometry.batch_size <= 0 ||
        geometry.resident_graph_rows <= 0 || profile.n_heads <= 0 ||
        profile.head_dim <= 0 || geometry.total_shards <= 0 ||
        (!has_exact_local_query_heads &&
         profile.n_heads % geometry.total_shards != 0))
    {
        throw std::runtime_error(
            "Attention workspace planning requires positive graph/device "
            "geometry, physical execution-unit count, and either an exact local query "
            "head assignment or a uniformly divisible model head count");
    }

    const int local_query_heads =
        has_exact_local_query_heads
            ? geometry.local_query_heads
            : profile.n_heads / geometry.total_shards;

    if (geometry.device.is_cpu())
    {
        const attention::AttentionWorkspaceCardinality cardinality =
            attention::planAttentionWorkspaceCardinality(
                geometry.resident_graph_rows,
                geometry.batch_size);
        return attention_workspace::cpuParallelRequirements({
                   .compact_query_rows = cardinality.compact_query_rows,
                   .local_query_heads = local_query_heads,
                   .head_dim = profile.head_dim,
                   .worker_count = geometry.device_compute_units,
               })
            .total_bytes_with_alignment();
    }
    if (!geometry.device.is_gpu())
    {
        throw std::runtime_error(
            "Attention workspace planning requires a CPU, CUDA, or ROCm device");
    }
    if (geometry.max_context_rows <= 0)
    {
        throw std::runtime_error(
            "GPU attention workspace planning requires a positive configured context horizon");
    }

    size_t partial_output = 0;
    size_t partial_m = 0;
    size_t partial_l = 0;

    if (geometry.device.is_cuda())
    {
        const cuda::fa2_policy::FA2PrefillParallelGeometry current{
            .batch_size = geometry.batch_size,
            .query_rows = geometry.resident_graph_rows,
            .local_query_heads = local_query_heads,
            .head_dim = profile.head_dim,
            .kv_capacity = geometry.max_context_rows,
            .sm_count = geometry.device_compute_units,
            .requested_axis =
                attention::AttentionPrefillParallelAxis::GeometrySelected,
        };
        const auto plan = cuda::fa2_policy::selectFA2PrefillParallelPlan(current);
        if (!plan.valid)
        {
            throw std::runtime_error(
                "CUDA attention workspace policy rejected admitted graph geometry");
        }
        partial_output = plan.partial_output_bytes;
        partial_m = plan.partial_m_bytes;
        partial_l = plan.partial_l_bytes;

        // Physical admission consumes the same non-monotonic family envelope
        // as AttentionComputeStage, before weights or captures own any bytes.
        // The admitted query-row bound is independent of the full KV horizon:
        // charging unadmitted larger buckets defeats resident-bucket selection.
        const auto family = cuda::fa2_policy::
            selectFA2GeometrySelectedWorkspaceEnvelope(current);
        if (family.usesContextParallelism())
        {
            partial_output = std::max(partial_output, family.partial_output_bytes);
            partial_m = std::max(partial_m, family.partial_m_bytes);
            partial_l = std::max(partial_l, family.partial_l_bytes);
        }
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

        // ROCm obeys the same admitted-query/full-context distinction as CUDA.
        const auto family = rocm::fa2_policy::
            selectROCmFA2GeometrySelectedWorkspaceEnvelope(current);
        if (family.valid && family.usesContextParallelism())
        {
            partial_output = std::max(
                partial_output, family.partial_output_bytes);
            partial_m = std::max(partial_m, family.partial_m_bytes);
            partial_l = std::max(partial_l, family.partial_l_bytes);
        }
    }

    const int local_kv_heads =
        geometry.local_kv_heads > 0
            ? geometry.local_kv_heads
            : std::max(
                  1,
                  (profile.n_kv_heads + geometry.total_shards - 1) /
                      geometry.total_shards);
    const attention::AttentionWorkspaceCardinality cardinality =
        attention::planAttentionWorkspaceCardinality(
            geometry.resident_graph_rows,
            geometry.batch_size);
    return attention_workspace::requirements({
               .compact_query_rows = cardinality.compact_query_rows,
               .request_count = cardinality.request_count,
               .local_query_heads = local_query_heads,
               .local_kv_heads = local_kv_heads,
               .head_dim = profile.head_dim,
               .context_rows = geometry.max_context_rows,
               .decode_splits =
                   attention_workspace::kMaximumDecodeSplits,
               .partial_output_floor_bytes = partial_output,
               .partial_m_floor_bytes = partial_m,
               .partial_l_floor_bytes = partial_l,
               // Activation precision is selected after metadata admission.
               // Reserve the union required by the FP32-compatible graph so a
               // later precision choice cannot invalidate physical capacity.
               .include_device_params = true,
               .include_fp32_kv_conversion = true,
           })
        .total_bytes_with_alignment();
}

/**
 * @brief Price the two graph-stable KV precision-conversion buffers.
 *
 * GPU KV caches publish one K and one V scratch address for the complete
 * context horizon. Every supported storage precision must be able to expose
 * FP32 rows, so four bytes per logical element is the common upper envelope
 * used by both CUDA and ROCm cache implementations. Multiple main/MTP caches
 * share these stable names and therefore retain the maximum pair once.
 */
size_t exactKVConversionWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (!geometry.device.is_gpu())
        return 0u;
    if (geometry.batch_size <= 0 || geometry.max_context_rows <= 0 ||
        profile.head_dim <= 0 || profile.n_kv_heads <= 0 ||
        geometry.total_shards <= 0)
    {
        throw std::runtime_error(
            "KV workspace planning requires positive batch, context, head, and TP geometry");
    }

    const int local_kv_heads =
        geometry.local_kv_heads > 0
            ? geometry.local_kv_heads
            : std::max(
                  1,
                  (profile.n_kv_heads + geometry.total_shards - 1) /
                      geometry.total_shards);
    const size_t local_kv_width = checkedMultiply(
        static_cast<size_t>(local_kv_heads),
        static_cast<size_t>(profile.head_dim),
        "KV conversion local width");
    const WorkspaceRequirements requirements =
        kv_cache_workspace::conversionRequirements({
            .configured_batch_size = geometry.batch_size,
            .configured_context_rows = geometry.max_context_rows,
            .requested_graph_rows = geometry.resident_graph_rows,
            .requested_batch_size = geometry.batch_size,
            .conversion_row_bytes = checkedMultiply(
                local_kv_width,
                sizeof(float),
                "KV conversion FP32 row"),
            .native_row_bytes = 0u,
        });
    return requirements.total_bytes_with_alignment();
}

/**
 * @brief Price the graph-stable RoPE publication ABI on GPU participants.
 *
 * RoPE owns one row-varying position buffer, an initialize-once bank of inverse
 * frequencies, and a device-owned replay parameter.  The same contract is
 * returned by every CUDA/ROCm precision specialization.
 */
size_t exactRoPEWorkspaceBytes(
    const ModelMemoryProfile& profile,
    const WorkspaceMemoryGeometry& geometry)
{
    if (!geometry.device.is_gpu() ||
        !hasFullAttentionLayer(
            profile, geometry.first_layer, geometry.last_layer))
    {
        return 0u;
    }
    return rope_workspace::requirements({
               .graph_rows = geometry.resident_graph_rows,
           })
        .total_bytes_with_alignment();
}

/**
 * @brief Price the graph-stable token-ID publication on its embedding owner.
 *
 * Pipeline followers do not build an embedding stage. Every participant that
 * does build one, including each vocabulary-parallel TP shard, publishes the
 * same retained row capacity through the backend-neutral embedding contract.
 */
size_t exactEmbeddingWorkspaceBytes(
    const WorkspaceMemoryGeometry& geometry)
{
    if (!geometry.device.is_gpu() || !geometry.owns_embedding)
        return 0u;
    return embedding_workspace::requirements({
               .graph_rows = geometry.resident_graph_rows,
           })
        .total_bytes_with_alignment();
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
    const size_t legacy_dense_baseline_bytes = estimate(
        geometry.batch_size,
        geometry.resident_graph_rows,
        profile.d_model,
        geometry.local_d_ff,
        profile.vocab_size,
        geometry.device);
    /*
     * ROCm quantized graphs can be dominated by the compact LM-head split-K
     * arena rather than prompt-row scratch.  Use the same stable-name factory
     * as the production kernel and retain the larger complete dense envelope;
     * the historical floor remains only for non-GEMM graph primitives that do
     * not yet expose typed metadata-only contracts.
     */
    const size_t rocm_quantized_gemm_bytes =
        exactROCmQuantizedGemmWorkspaceBytes(profile, geometry);
    /*
     * A complete ROCm quantized matrix inventory supplies the real stable-name
     * GEMM contract. The historical 800 MiB blanket predates that inventory
     * and overlaps the exact MoE, attention, and GEMM names added below; using
     * both creates a second physical ledger. Profiles without such an
     * inventory retain the legacy conservative envelope until their backend
     * exposes equivalent metadata-only contracts.
     */
    const size_t dense_baseline_bytes =
        geometry.device.is_rocm() && rocm_quantized_gemm_bytes > 0u
            ? 0u
            : legacy_dense_baseline_bytes;
    size_t bytes = std::max(
        dense_baseline_bytes, rocm_quantized_gemm_bytes);

    const size_t moe_bytes = geometry.device.is_gpu()
        ? exactMoEWorkspaceBytes(
              profile,
              geometry.batch_size,
              geometry.resident_graph_rows,
              geometry.device)
        : 0u;
    bytes = checkedAdd(
        bytes,
        moe_bytes,
        "MoE graph-family requirements");

    const size_t attention_bytes =
        exactAttentionWorkspaceBytes(profile, geometry);
    bytes = checkedAdd(
        bytes,
        attention_bytes,
        "attention context-summary requirements");

    const size_t kv_conversion_bytes =
        exactKVConversionWorkspaceBytes(profile, geometry);
    bytes = checkedAdd(
        bytes,
        kv_conversion_bytes,
        "KV precision-conversion requirements");

    const size_t rope_bytes = exactRoPEWorkspaceBytes(profile, geometry);
    bytes = checkedAdd(
        bytes,
        rope_bytes,
        "RoPE graph-stable publication requirements");

    const size_t embedding_bytes = exactEmbeddingWorkspaceBytes(geometry);
    bytes = checkedAdd(
        bytes,
        embedding_bytes,
        "embedding token publication requirements");

    const size_t execution_rows = checkedMultiply(
        static_cast<size_t>(std::max(1, geometry.batch_size)),
        static_cast<size_t>(std::max(1, geometry.resident_graph_rows)),
        "floating projection execution rows");
    const size_t floating_projection_bytes =
        gpuFloatingPointWorkspaceBytes(
            profile, geometry, execution_rows, /*routed_only=*/false);
    bytes = checkedAdd(
        bytes,
        floating_projection_bytes,
        "GPU floating projection requirements");

    size_t hybrid_row_scratch_bytes = 0u;
    if (geometry.device.is_gpu() &&
        hasHybridRecurrentLayer(
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
            hybrid_row_scratch_bytes =
                checkedMultiply(execution_rows, hybrid_row_bytes,
                                "hybrid recurrent row scratch");
            bytes = checkedAdd(
                bytes,
                hybrid_row_scratch_bytes,
                "hybrid recurrent scratch");
        }
    }

    size_t compact_family_bytes = 0u;
    size_t persistent_hybrid_snapshot_bytes = 0u;
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
        compact_family_bytes = estimate(profile, mtp_geometry);
        persistent_hybrid_snapshot_bytes =
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

    logVramBomLine(
        "workspace_admission_estimate",
        "device=" + geometry.device.toString() +
            " resident_graph_rows=" +
            std::to_string(geometry.resident_graph_rows) +
            " mtp_target_query_rows=" +
            std::to_string(geometry.mtp_target_query_rows) +
            " dense_baseline_bytes=" +
            std::to_string(dense_baseline_bytes) +
            " rocm_quantized_gemm_bytes=" +
            std::to_string(rocm_quantized_gemm_bytes) +
            " moe_bytes=" + std::to_string(moe_bytes) +
            " attention_bytes=" + std::to_string(attention_bytes) +
            " kv_conversion_bytes=" +
            std::to_string(kv_conversion_bytes) +
            " rope_bytes=" + std::to_string(rope_bytes) +
            " floating_projection_bytes=" +
            std::to_string(floating_projection_bytes) +
            " hybrid_row_scratch_bytes=" +
            std::to_string(hybrid_row_scratch_bytes) +
            " compact_family_bytes=" +
            std::to_string(compact_family_bytes) +
            " persistent_hybrid_snapshot_bytes=" +
            std::to_string(persistent_hybrid_snapshot_bytes) +
            " admitted_bytes=" + std::to_string(bytes));

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

        /*
         * The MoE requirement factory owns routing/grouping scratch. Prepared
         * GEMM kernels own a separate graph ABI and are discovered through
         * IWorkspaceConsumer at runtime. Compose that same codebook-neutral
         * ABI here before expert weights consume the remaining physical
         * capacity; a late allocator expansion is intentionally impossible.
         */
        const WorkspaceRequirements gemm_requirements =
            rocmRoutedExpertGemmRequirements(
                profile, geometry, direct_rows, compact_rows);
        direct_requirements.merge(gemm_requirements);
    }
    direct_requirements.merge(compact_requirements);
    size_t bytes = direct_requirements.total_bytes_with_alignment();
    WorkspaceMemoryGeometry expert_geometry = geometry;
    expert_geometry.apportioned_routed_experts = true;
    bytes = checkedAdd(
        bytes,
        gpuFloatingPointWorkspaceBytes(
            profile,
            expert_geometry,
            static_cast<size_t>(compact_rows),
            /*routed_only=*/true),
        "CUDA local floating-expert requirements");
    logVramBomLine(
        "workspace_routed_expert_admission_estimate",
        "device=" + geometry.device.toString() +
            " direct_rows=" + std::to_string(direct_rows) +
            " compact_rows=" + std::to_string(compact_rows) +
            " buffers=" +
            std::to_string(direct_requirements.buffers.size()) +
            " admitted_bytes=" + std::to_string(bytes));
    return bytes;
}

} // namespace llaminar2
