/**
 * @file ActivationMemoryEstimator.cpp
 * @brief Implements exact captured graph-arena byte accounting.
 *
 * BufferArena currently allocates each registered BufferId independently;
 * alias metadata validates lifetimes but does not merge physical storage.
 * CPU admission resolves the production GraphSchema into the same root inventory
 * consumed by graph setup. CPUTensorStorageDeclaration owns native extents and
 * allocator rounding. GPU estimates retain their existing geometry policy.
 */

#include "planning/ActivationMemoryEstimator.h"
#include "tensors/AlignedVector.h"
#include <initializer_list>
#include "kernels/gdn/GDNPreparedQK.h"
#include "execution/mtp/GenerationRequestSeeds.h"
#include "execution/mtp/MTPTerminalGatherGeometry.h"
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "models/qwen/Qwen2Schema.h"
#include "config/GDNHeadAssignment.h"
#include "memory/GraphArenaStorageLayout.h"
#include "execution/local_execution/graph/GraphResolver.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{

namespace
{

/** @brief Add two admission terms without permitting size_t wraparound. */
size_t checkedAdd(size_t left, size_t right, std::string_view contribution)
{
    if (right > std::numeric_limits<size_t>::max() - left)
    {
        throw std::overflow_error(
            "Activation memory overflow while adding " +
            std::string(contribution));
    }
    return left + right;
}

/** @brief Multiply two admission terms without permitting size_t wraparound. */
size_t checkedMultiply(size_t left, size_t right, std::string_view contribution)
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    {
        throw std::overflow_error(
            "Activation memory overflow while multiplying " +
            std::string(contribution));
    }
    return left * right;
}

/** @brief One checked QFN resolver projection shared by base and concrete tier estimates. */
std::pair<GraphResolverConfig, Qwen4ExpBufferDimensions> qwenLookupGeometry(
    const ModelMemoryProfile &profile, const ActivationGraphMemoryGeometry &geometry, DeviceId device)
{
    if (!profile.flash_next || geometry.batch_size <= 0 || geometry.resident_graph_rows <= 0 ||
        geometry.mtp_target_query_rows <= 0)
        throw std::invalid_argument("QFN lookup geometry requires exact metadata and positive admitted rows");
    if (geometry.context_tokens <= 0)
        throw std::invalid_argument("Flash Next activation admission requires explicit runtime context capacity");
    const auto flattened = checkedMultiply(static_cast<std::size_t>(geometry.batch_size),
        static_cast<std::size_t>(geometry.resident_graph_rows), "Flash Next flattened graph rows");
    if (flattened > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Flash Next flattened graph rows exceed resolver capacity");
    GraphResolverConfig resolver;
    resolver.seq_len = static_cast<int>(flattened);
    resolver.d_model = profile.d_model;
    resolver.local_n_heads = geometry.local_n_heads;
    resolver.default_device = device;
    resolver.custom_formulas["mtp_kv_prefill_rows"] = std::max(flattened, static_cast<std::size_t>(geometry.mtp_target_query_rows));
    resolver.custom_formulas["mtp_target_query_rows"] = static_cast<std::size_t>(geometry.mtp_target_query_rows);
    resolver.custom_formulas["terminal_logit_row_capacity"] = static_cast<size_t>(
        std::max(geometry.batch_size, geometry.mtp_target_query_rows));
    const Qwen4ExpBufferDimensions dimensions{
        .main_layers = profile.n_layers, .first_layer = geometry.first_layer,
        .layer_count = (geometry.last_layer < 0 ? profile.n_layers - 1 : geometry.last_layer) - geometry.first_layer + 1,
        .compute_width = profile.d_model, .context_tokens = geometry.context_tokens,
        // Sparse request banks follow GraphConfig.max_request_count, which
        // the production factory projects from ordinary runtime batch size.
        // MTP request/seed capacity is a separate retained storage contract.
        .request_capacity = geometry.batch_size,
        .flattened_rows = static_cast<int>(flattened), .local_query_heads = geometry.local_n_heads,
        .retained_mtp_candidate_rows = geometry.retained_mtp_candidate_rows,
        .device = device, .attention_phase_schedule=geometry.attention_phase_schedule, .ngram_lookup = geometry.ngram_lookup,
        .ngram_fixed_head_participants = geometry.ngram_fixed_head_participants};
    return {std::move(resolver), dimensions};
}

/**
 * @brief Price independently allocated FP32 roots using native allocator rounding.
 * @param rows Shared row capacity of these roots.
 * @param widths Column capacity for each distinct physical root, not a concatenated bank.
 * @param device Allocator owner; CPU storage follows AlignedVector's physical contract.
 * @param contribution Diagnostic name for checked geometry arithmetic.
 * @return Sum of exact root extents; GPU retains its native unrounded tensor extent.
 */
size_t fp32RootBytes(size_t rows, std::initializer_list<size_t> widths,
    DeviceId device, std::string_view contribution)
{
    size_t bytes = 0;
    for (const auto width : widths)
    {
        const auto elements = checkedMultiply(rows, width, contribution);
        const auto root_bytes = device.is_cpu()
            ? AlignedVector<float>::requiredAllocationBytes(elements)
            : checkedMultiply(elements, sizeof(float), contribution);
        bytes = checkedAdd(bytes, root_bytes, contribution);
    }
    return bytes;
}

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

size_t projectionInputRows(
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
            tensor.name.find(name_fragment) == std::string::npos)
        {
            continue;
        }
        maximum = std::max(maximum, tensor.K);
    }
    return maximum;
}

size_t shardColumns(size_t columns, int total_shards)
{
    const size_t shards = static_cast<size_t>(std::max(1, total_shards));
    return columns / shards + (columns % shards != 0 ? 1u : 0u);
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

} // namespace

GraphArenaStorageLayout ActivationMemoryEstimator::declareCPUStorage(
    const ModelMemoryProfile &profile, const ActivationGraphMemoryGeometry &geometry, DeviceId device)
{
    if (!device.is_valid() || !device.is_cpu() || geometry.batch_size <= 0 ||
        geometry.resident_graph_rows <= 0 || geometry.total_shards <= 0 ||
        geometry.mtp_target_query_rows <= 0)
        throw std::invalid_argument("CPU graph storage requires exact allocator and positive admitted geometry");
    const size_t rows = checkedMultiply(static_cast<size_t>(geometry.batch_size),
        static_cast<size_t>(geometry.resident_graph_rows), "CPU flattened graph rows");
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("CPU graph rows exceed resolver representation");

    // This adapter projects source metadata into the graph resolver. Physical
    // root names, shapes, placement and dtypes remain exclusively in schemas;
    // no root count or byte formula is reconstructed by the estimator.
    GraphResolverConfig resolver;
    resolver.default_device = device;
    resolver.seq_len = static_cast<int>(rows);
    resolver.batch_size = 1;
    resolver.n_layers = profile.n_layers;
    resolver.d_model = profile.d_model;
    resolver.d_ff = profile.d_ff;
    resolver.n_heads = profile.n_heads;
    resolver.n_kv_heads = profile.n_kv_heads;
    resolver.head_dim = profile.head_dim;
    resolver.vocab_size = profile.vocab_size;
    resolver.local_d_ff = geometry.local_d_ff;
    resolver.local_n_heads = geometry.local_n_heads;
    resolver.local_n_kv_heads = geometry.local_n_kv_heads;
    resolver.local_vocab = geometry.local_vocab > 0 ? geometry.local_vocab
        : static_cast<int>(shardColumns(static_cast<size_t>(profile.vocab_size), geometry.total_shards));
    auto &formulas = resolver.custom_formulas;
    const size_t target_rows = static_cast<size_t>(geometry.mtp_target_query_rows);
    const size_t query_columns = checkedMultiply(static_cast<size_t>(geometry.local_n_heads),
        static_cast<size_t>(profile.head_dim), "CPU query columns");
    const size_t kv_columns = checkedMultiply(static_cast<size_t>(geometry.local_n_kv_heads),
        static_cast<size_t>(profile.head_dim), "CPU KV columns");
    formulas["mtp_target_query_rows"] = target_rows;
    // Ordinary compact terminal rows follow GraphConfig.max_request_count's
    // production batch projection, independently of MTP request/seed capacity.
    formulas["terminal_logit_row_capacity"] = std::max(static_cast<size_t>(geometry.batch_size), target_rows);
    formulas["mtp_kv_prefill_rows"] = std::max(rows, target_rows);
    const bool replicated_predictor = geometry.mtp_shifted_kv_head_layout ==
        MTPShiftedKVHeadLayout::FullModelPerParticipant;
    formulas["mtp_q_dim"] = replicated_predictor ? checkedMultiply(
        static_cast<size_t>(profile.n_heads), static_cast<size_t>(profile.head_dim), "CPU replicated predictor Q") : query_columns;
    formulas["mtp_kv_dim"] = replicated_predictor ? checkedMultiply(
        static_cast<size_t>(profile.n_kv_heads), static_cast<size_t>(profile.head_dim), "CPU replicated predictor KV") : kv_columns;
    formulas["mtp_d_ff"] = static_cast<size_t>(replicated_predictor ? profile.d_ff : geometry.local_d_ff);
    formulas["mtp_vocab"] = geometry.mtp_terminal_logits_layout == MTPTerminalLogitsLayout::FullVocabularyPerParticipant
        ? static_cast<size_t>(profile.vocab_size) : static_cast<size_t>(resolver.local_vocab);
    const auto gather = MTPTerminalGatherGeometry::resolve(geometry.mtp_terminal_logits_layout,
        target_rows, static_cast<size_t>(profile.vocab_size));
    formulas["mtp_gather_rows"] = gather.rows();
    formulas["mtp_gather_vocab"] = gather.columns();

    const bool hybrid = profile.flash_next || hasHybridRecurrentLayer(profile, 0, std::max(0, profile.n_layers - 1));
    GraphSchema schema = hybrid ? (profile.expert_count > 0
        ? Qwen35MoESchemaFactory{}.createSchema() : Qwen35SchemaFactory{}.createSchema())
        : Qwen2SchemaFactory{}.createSchema();
    if (hybrid)
    {
        if (profile.gdn_state_size <= 0 || profile.gdn_group_count <= 0 ||
            profile.gdn_time_step_rank <= 0 || profile.gdn_inner_size <= 0)
            throw std::invalid_argument("CPU hybrid arena requires complete source GDN metadata");
        const auto heads = GDNHeadAssignment::fromPartition(profile.gdn_group_count,
            profile.gdn_time_step_rank, 0, geometry.local_n_heads, profile.n_heads);
        const size_t inner = checkedMultiply(static_cast<size_t>(profile.gdn_inner_size),
            static_cast<size_t>(heads.localValueHeads()), "CPU local GDN width") /
            static_cast<size_t>(profile.gdn_time_step_rank);
        const size_t keys = checkedMultiply(static_cast<size_t>(heads.localKeyHeads()),
            static_cast<size_t>(profile.gdn_state_size), "CPU local GDN keys");
        formulas["gdn_inner_size"] = inner;
        formulas["gdn_qkv_dim"] = checkedAdd(checkedMultiply(2, keys, "CPU GDN Q/K"), inner, "CPU GDN QKV");
        formulas["gdn_time_step_rank"] = static_cast<size_t>(heads.localValueHeads());
        formulas["fa_q_full_dim"] = checkedMultiply(2, query_columns, "CPU FA query/gate");
        formulas["mtp_fa_q_full_dim"] = checkedMultiply(2, formulas.at("mtp_q_dim"), "CPU predictor query/gate");
        formulas["attn_output_dim"] = std::max(query_columns, inner);
        formulas["mtp_attn_output_dim"] = std::max(formulas.at("mtp_q_dim"), formulas.at("attn_output_dim"));
    }
    if (hybrid && profile.expert_count > 0)
    {
        if (geometry.routed_compute_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
            throw std::invalid_argument("MoE projection arena requires a native GPU domain");
        formulas["moe_top_k"] = static_cast<size_t>(profile.expert_used_count);
        formulas["moe_canonical_publication_slots"] = checkedAdd(
            static_cast<size_t>(profile.expert_used_count), static_cast<size_t>(geometry.total_shards),
            "CPU MoE publication slots");
        formulas["moe_expert_intermediate"] = static_cast<size_t>(profile.expert_feed_forward_length);
        formulas["moe_ffn_intermediate_max"] = static_cast<size_t>(std::max(
            profile.expert_feed_forward_length, profile.expert_shared_feed_forward_length));
        formulas["moe_activation_rows"] = std::max(rows,
            geometry.retained_mtp_candidate_rows > 0 ? target_rows : size_t{1});
    }
    if (profile.flash_next)
    {
        const auto [unused, dimensions] = qwenLookupGeometry(profile, geometry, device);
        applyQwen4ExpBufferLayout(schema, *profile.flash_next, dimensions);
        applyQwen4ExpBufferGeometry(resolver, *profile.flash_next, dimensions);
    }
    const bool retained_mtp = geometry.retained_mtp_candidate_rows > 0;
    return GraphArenaStorageLayout::resolve(schema, resolver, {
        .full_prefill_kv_handoff = geometry.replicated_dense_decode,
        .terminal_hidden_rows = retained_mtp ? std::max(static_cast<size_t>(geometry.batch_size), target_rows) : 0,
        .terminal_hidden_columns = retained_mtp ? (profile.flash_next
            ? static_cast<size_t>(profile.flash_next->residual_width) : static_cast<size_t>(profile.d_model)) : 0});
}

size_t ActivationMemoryEstimator::estimate(
    int batch_size,
    int max_seq_len,
    int d_model,
    int d_ff,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int vocab_size,
    DeviceId device)
{
    if (batch_size <= 0 || max_seq_len <= 0 || d_model <= 0)
    {
        return 0;
    }

    const size_t B = static_cast<size_t>(batch_size);
    const size_t S = static_cast<size_t>(max_seq_len);
    const size_t D = static_cast<size_t>(std::max(0, d_model));
    const size_t F = static_cast<size_t>(std::max(0, d_ff));
    const size_t H = static_cast<size_t>(std::max(0, n_heads));
    const size_t HK = static_cast<size_t>(std::max(0, n_kv_heads));
    const size_t HD = static_cast<size_t>(std::max(0, head_dim));
    const size_t V = static_cast<size_t>(std::max(0, vocab_size));


    // All sizes in bytes (FP32 = 4 bytes per element unless noted)
    const size_t Q = checkedMultiply(H, HD, "query columns");
    const size_t KV = checkedMultiply(HK, HD, "KV columns");

    /*
     * BufferArena currently gives every registered BufferId physical storage.
     * Alias-group metadata validates borrowing but does not merge allocations.
     * Therefore planning must count the logical owner set, not a hypothetical
     * phase peak:
     *
     *   hidden, normalized, residual, attention projection, FFN output = 5D
     *   Q and attention output                                      = 2Q
     *   K and V                                                     = 2KV
     *   gate and up                                                 = 2F
     *   causal-mask/GEMM workspace                                  = S
     *
     * Graphs with different row buckets use these same BufferIds serially, so
     * this is one largest-participant arena rather than one arena per graph.
     */
    const size_t resident_rows = checkedMultiply(B, S, "resident graph rows");
    const auto resident_buffers = fp32RootBytes(resident_rows,
        {D, D, D, D, D, Q, Q, KV, KV, F, F, S}, device, "common graph roots");
    return checkedAdd(resident_buffers,
        fp32RootBytes(B, {V}, device, "terminal logits"), "terminal logits");
}

size_t ActivationMemoryEstimator::estimate(
    const ModelMemoryProfile& profile,
    const ActivationGraphMemoryGeometry& geometry,
    DeviceId device)
{
    if (geometry.batch_size <= 0 || geometry.resident_graph_rows <= 0 ||
        geometry.local_d_ff < 0 || geometry.local_n_heads < 0 ||
        geometry.local_n_kv_heads < 0 || geometry.local_vocab < 0 ||
        geometry.total_shards <= 0 ||
        geometry.mtp_target_query_rows <= 0 || geometry.generation_request_capacity <= 0)
    {
        throw std::invalid_argument(
            "Activation graph memory geometry requires positive rows, batch, "
            "shards, and MTP capacity with non-negative local widths");
    }

    if (device.is_cpu())
        return declareCPUStorage(profile, geometry, device).cpuPhysicalBytes();

    size_t bytes = estimate(
        geometry.batch_size,
        geometry.resident_graph_rows,
        profile.d_model,
        geometry.local_d_ff,
        geometry.local_n_heads,
        geometry.local_n_kv_heads,
        profile.head_dim,
        profile.vocab_size,
        device);
    if (bytes == 0)
        return 0;

    // This request input exists on GPU even with no predictor. Use the same
    // geometry as arena registration; CPU draws do not retain device seeds.
    if (device.is_gpu())
        bytes = checkedAdd(bytes, GenerationRequestSeedGeometry(
            geometry.batch_size, geometry.generation_request_capacity).bytes(),
            "resident generation request seeds");

    const size_t rows = static_cast<size_t>(geometry.resident_graph_rows);
    const size_t target_rows =
        static_cast<size_t>(geometry.mtp_target_query_rows);
    const size_t mtp_prefill_rows = std::max(rows, target_rows);
    const size_t d_model = static_cast<size_t>(std::max(0, profile.d_model));
    const size_t local_d_ff =
        static_cast<size_t>(geometry.local_d_ff);
    const size_t local_q = checkedMultiply(
        static_cast<size_t>(geometry.local_n_heads),
        static_cast<size_t>(std::max(0, profile.head_dim)),
        "local query columns");
    const size_t local_kv = checkedMultiply(
        static_cast<size_t>(geometry.local_n_kv_heads),
        static_cast<size_t>(std::max(0, profile.head_dim)),
        "local KV columns");
    const size_t full_kv = checkedMultiply(
        static_cast<size_t>(std::max(0, profile.n_kv_heads)),
        static_cast<size_t>(std::max(0, profile.head_dim)),
        "full KV columns");
    const size_t local_vocab = geometry.local_vocab > 0
        ? static_cast<size_t>(geometry.local_vocab)
        : shardColumns(
              static_cast<size_t>(std::max(0, profile.vocab_size)),
              geometry.total_shards);

    if (geometry.replicated_dense_decode && device.is_gpu())
    {
        // The prefill graph writes TP-local K/V, then its indexed CPU or
        // captured native GPU handoff publishes full replicated decode rows.
        // These two BufferIds coexist with the ordinary K/V arena owners.
        bytes = checkedAdd(
            bytes,
            fp32RootBytes(rows, {full_kv, full_kv}, device, "phase-split full KV handoff"),
            "phase-split full KV handoff");
    }

    /* The model schema registers a participant-local terminal-logit shard. */
    bytes = checkedAdd(
        bytes,
        fp32RootBytes(1u, {local_vocab}, device, "local terminal logits"),
        "local terminal logits");

    /*
     * Qwen3.5/3.6 hybrid layers add physical BufferIds beyond the common
     * transformer schema. Infer their exact row widths from projection tensor
     * geometry so planning follows the model file rather than a model-name
     * heuristic. GraphSchema registers the union for every continuation
     * participant, even a pipeline shard whose assigned layers happen to be
     * full-attention-only, so inspect the complete tensor inventory here.
     */
    const int inventory_first_layer = 0;
    const int inventory_last_layer = std::max(0, profile.n_layers - 1);
    if (!hasHybridRecurrentLayer(
            profile, inventory_first_layer, inventory_last_layer))
    {
        return bytes;
    }

    const int schema_shards =
        geometry.replicated_dense_decode ? 1 : geometry.total_shards;
    const size_t full_attention_q = shardColumns(
        projectionOutputRows(
            profile,
            ".attn_q.weight",
            inventory_first_layer,
            inventory_last_layer),
        schema_shards);
    const size_t full_attention_gate = shardColumns(
        projectionOutputRows(
            profile,
            ".attn_gate.weight",
            inventory_first_layer,
            inventory_last_layer),
        schema_shards);
    const size_t gdn_qkv = shardColumns(
        projectionOutputRows(
            profile,
            ".attn_qkv.weight",
            inventory_first_layer,
            inventory_last_layer),
        schema_shards);
    const size_t gdn_z = shardColumns(
        projectionInputRows(
            profile,
            ".ssm_out.weight",
            inventory_first_layer,
            inventory_last_layer),
        schema_shards);
    const size_t gdn_alpha = shardColumns(
        projectionOutputRows(
            profile,
            ".ssm_alpha.weight",
            inventory_first_layer,
            inventory_last_layer),
        schema_shards);
    const size_t gdn_beta = shardColumns(
        projectionOutputRows(
            profile,
            ".ssm_beta.weight",
            inventory_first_layer,
            inventory_last_layer),
        schema_shards);

    if (full_attention_q == 0 || full_attention_gate == 0 ||
        gdn_qkv == 0 || gdn_z == 0)
    {
        throw std::invalid_argument(
            "Hybrid graph-arena planning requires complete FA and GDN "
            "projection geometry in the GGUF tensor inventory");
    }

    const size_t attention_output = std::max(local_q, gdn_z);
    if (attention_output > local_q)
    {
        bytes = checkedAdd(
            bytes,
            fp32RootBytes(rows, {attention_output}, device, "hybrid attention-output root") -
                fp32RootBytes(rows, {local_q}, device, "base attention-output root"),
            "hybrid attention-output width delta");
    }

    // The scalar prepared program adds one arena activation, not a row-capacity
    // multiple or per-layer state bank. Schema and BOM share this exact layout.
    if (device.is_rocm() && gdn::ScalarPreparedQKLayout::supportsNativeGeometry(
            profile.gdn_state_size, profile.gdn_conv_kernel_size))
    {
        const int key_heads = profile.gdn_group_count > 0
            ? profile.gdn_group_count : profile.n_heads;
        const int value_heads = profile.gdn_time_step_rank > 0
            ? profile.gdn_time_step_rank : key_heads;
        // The replicated-decode schema union owns full-width buffers, even
        // when its prefill graph retains a smaller actual TP interval.
        const int query_heads = geometry.replicated_dense_decode
            ? profile.n_heads : geometry.local_n_heads;
        bytes = checkedAdd(bytes, gdn::ScalarPreparedQKLayout::fromHeadPartition(
            key_heads, value_heads, profile.gdn_state_size,
            query_heads, profile.n_heads).bytes, "scalar GDN prepared Q/K activation");
    }

    /* gdn_qkv and gdn_recurrence_in are distinct simultaneous owners. */
    bytes = checkedAdd(bytes, fp32RootBytes(rows,
        {full_attention_q, full_attention_gate, gdn_qkv, gdn_qkv, gdn_z, gdn_alpha, gdn_beta},
        device, "hybrid graph roots"), "hybrid graph roots");

    /*
     * Qwen3.5 registers one complete MTP sidecar union. Shifted prefill reuses
     * the resident bucket width, while verifier-only outputs use the flattened
     * target-row capacity. These addresses exist even before the first MTP
     * transaction and therefore belong in fixed admission.
     */
    bytes = checkedAdd(bytes, fp32RootBytes(mtp_prefill_rows,
        {d_model, d_model, d_model, checkedMultiply(2u, d_model, "MTP concat"), d_model,
         full_attention_q, local_q, local_q, local_kv, local_kv, full_kv, full_kv},
        device, "MTP shifted-prefill roots"), "MTP shifted-prefill roots");

    const size_t mtp_vocab =
        geometry.mtp_terminal_logits_layout ==
                MTPTerminalLogitsLayout::FullVocabularyPerParticipant
            ? static_cast<size_t>(std::max(0, profile.vocab_size))
            : local_vocab;
    bytes = checkedAdd(bytes, fp32RootBytes(target_rows,
        {d_model, d_model, d_model, attention_output, local_d_ff, local_d_ff, mtp_vocab},
        device, "MTP target-row roots"), "MTP target-row roots");

    /* Stable selected-row input and the exact retained terminal output bank. */
    bytes = checkedAdd(
        bytes,
        fp32RootBytes(1u, {d_model}, device, "LM-head selected row"),
        "LM-head selected row");
    bytes = checkedAdd(
        bytes,
        fp32RootBytes(std::max(static_cast<size_t>(geometry.batch_size), target_rows),
            {d_model}, device, "LM-head compact terminal rows"),
        "LM-head compact terminal rows");
    const auto mtp_gather = MTPTerminalGatherGeometry::resolve(
        geometry.mtp_terminal_logits_layout, target_rows,
        static_cast<size_t>(profile.vocab_size));
    bytes = checkedAdd(bytes,
        fp32RootBytes(mtp_gather.rows(), {mtp_gather.columns()}, device, "MTP gathered logits"),
        "MTP gathered logits");

    if (profile.expert_count > 0)
    {
        if (profile.expert_used_count <= 0 ||
            profile.expert_feed_forward_length <= 0)
        {
            throw std::invalid_argument(
                "MoE graph-arena planning requires positive top-k and expert "
                "intermediate geometry");
        }
        const size_t top_k =
            static_cast<size_t>(profile.expert_used_count);
        const size_t canonical_participants =
            static_cast<size_t>(std::max(1, geometry.total_shards));
        const size_t max_expert_intermediate = static_cast<size_t>(std::max(
            profile.expert_feed_forward_length,
            profile.expert_shared_feed_forward_length));

        bytes = checkedAdd(bytes, fp32RootBytes(mtp_prefill_rows,
            {top_k, top_k, d_model, d_model, max_expert_intermediate, max_expert_intermediate},
            device, "MoE independent roots"), "MoE independent roots");
        if (geometry.routed_compute_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
        {
            if (!device.is_gpu())
                throw std::invalid_argument("MoE projection arena requires a native GPU domain");
            bytes = checkedAdd(bytes,
                MoEProjectionArenaGeometry::resolve(profile, geometry.total_shards).bytes(mtp_prefill_rows),
                "MoE projection transaction arena");
        }
        else
        {
            const size_t slots = checkedAdd(top_k, canonical_participants, "MoE canonical publication slots");
            bytes = checkedAdd(bytes, fp32RootBytes(mtp_prefill_rows,
                {checkedMultiply(slots, d_model, "MoE canonical columns")},
                device, "MoE canonical root"), "MoE canonical root");
        }
    }
    if (profile.flash_next)
    {
        const auto [resolver, dimensions] = qwenLookupGeometry(profile, geometry, device);
        const Qwen35MoESchemaFactory common_factory;
        bytes = checkedAdd(bytes, qwen4ExpArenaExtensionBytes(common_factory.createSchema(), resolver,
            *profile.flash_next, dimensions), "Flash Next declared arena/state extensions");
    }
    return bytes;
}

std::size_t ActivationMemoryEstimator::immutableLookupArenaBytes(const ModelMemoryProfile &profile,
    const ActivationGraphMemoryGeometry &geometry, DeviceId device)
{
    if (!geometry.ngram_lookup)
        throw std::invalid_argument("Ngram execution contract is required for lookup arena candidate projection");
    const auto [resolver, dimensions] = qwenLookupGeometry(profile, geometry, device);
    return qwen4ExpLookupArenaBytes(resolver, *profile.flash_next, dimensions);
}

} // namespace llaminar2
