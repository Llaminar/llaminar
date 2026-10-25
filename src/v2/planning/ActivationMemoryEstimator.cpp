/**
 * @file ActivationMemoryEstimator.cpp
 * @brief Implements exact captured graph-arena byte accounting.
 *
 * BufferArena currently allocates each registered BufferId independently;
 * alias metadata validates lifetimes but does not merge physical storage.
 * Consequently this implementation sums the same declarative owner union used
 * by Qwen graph construction instead of estimating a phase peak.
 */

#include "planning/ActivationMemoryEstimator.h"
#include "kernels/gdn/GDNPreparedQK.h"
#include "execution/mtp/GenerationRequestSeeds.h"
#include "execution/mtp/MTPTerminalGatherGeometry.h"
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
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
    const Qwen4ExpBufferDimensions dimensions{
        .main_layers = profile.n_layers, .first_layer = geometry.first_layer,
        .layer_count = (geometry.last_layer < 0 ? profile.n_layers - 1 : geometry.last_layer) - geometry.first_layer + 1,
        .compute_width = profile.d_model, .context_tokens = geometry.context_tokens,
        .request_capacity = geometry.generation_request_capacity,
        .flattened_rows = static_cast<int>(flattened), .local_query_heads = geometry.local_n_heads,
        .retained_mtp_candidate_rows = geometry.retained_mtp_candidate_rows,
        .device = device, .attention_phase_schedule=geometry.attention_phase_schedule, .ngram_lookup = geometry.ngram_lookup,
        .ngram_fixed_head_participants = geometry.ngram_fixed_head_participants};
    return {std::move(resolver), dimensions};
}

/** @brief Return bytes for one FP32 row bank with checked arithmetic. */
size_t fp32RowBankBytes(
    size_t rows,
    size_t columns,
    std::string_view contribution)
{
    const size_t elements = checkedMultiply(rows, columns, contribution);
    return checkedMultiply(elements, sizeof(float), contribution);
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
    (void)device;

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
    size_t resident_row_elements = checkedMultiply(5u, D, "common D columns");
    resident_row_elements = checkedAdd(
        resident_row_elements,
        checkedMultiply(2u, Q, "common Q columns"),
        "common Q columns");
    resident_row_elements = checkedAdd(
        resident_row_elements,
        checkedMultiply(2u, KV, "common KV columns"),
        "common KV columns");
    resident_row_elements = checkedAdd(
        resident_row_elements,
        checkedMultiply(2u, F, "common FFN columns"),
        "common FFN columns");
    resident_row_elements = checkedAdd(
        resident_row_elements, S, "causal-mask columns");
    const size_t resident_rows = checkedMultiply(B, S, "resident graph rows");
    const size_t resident_buffers = fp32RowBankBytes(
        resident_rows, resident_row_elements, "common graph arena");

    // Ordinary prefill/decode computes only the selected terminal logits row.
    const size_t terminal_logits = fp32RowBankBytes(
        B, V, "terminal logits");
    return checkedAdd(resident_buffers, terminal_logits, "terminal logits");
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
        // The prefill graph writes TP-local K/V, then a captured handoff
        // publishes separate full-width rows to the replicated decode bank.
        // These two BufferIds coexist with the ordinary K/V arena owners.
        bytes = checkedAdd(
            bytes,
            fp32RowBankBytes(
                rows,
                checkedMultiply(2u, full_kv, "phase-split full KV handoff"),
                "phase-split full KV handoff"),
            "phase-split full KV handoff");
    }

    /* The model schema registers a participant-local terminal-logit shard. */
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(1u, local_vocab, "local terminal logits"),
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
            fp32RowBankBytes(
                rows,
                attention_output - local_q,
                "hybrid attention-output width delta"),
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
    const size_t hybrid_columns =
        full_attention_q +
        full_attention_gate +
        checkedMultiply(2u, gdn_qkv, "GDN projection and recurrence columns") +
        gdn_z +
        gdn_alpha +
        gdn_beta;
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(rows, hybrid_columns, "hybrid graph arena"),
        "hybrid graph arena");

    /*
     * Qwen3.5 registers one complete MTP sidecar union. Shifted prefill reuses
     * the resident bucket width, while verifier-only outputs use the flattened
     * target-row capacity. These addresses exist even before the first MTP
     * transaction and therefore belong in fixed admission.
     */
    size_t mtp_prefill_columns = checkedMultiply(
        6u, d_model, "MTP hidden/concat columns");
    mtp_prefill_columns = checkedAdd(
        mtp_prefill_columns, full_attention_q, "MTP raw-Q columns");
    mtp_prefill_columns = checkedAdd(
        mtp_prefill_columns,
        checkedMultiply(2u, local_q, "MTP query/gate columns"),
        "MTP query/gate columns");
    mtp_prefill_columns = checkedAdd(
        mtp_prefill_columns,
        checkedMultiply(2u, local_kv, "MTP local KV columns"),
        "MTP local KV columns");
    mtp_prefill_columns = checkedAdd(
        mtp_prefill_columns,
        checkedMultiply(2u, full_kv, "MTP full KV handoff columns"),
        "MTP full KV handoff columns");
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(
            mtp_prefill_rows,
            mtp_prefill_columns,
            "MTP shifted-prefill arena"),
        "MTP shifted-prefill arena");

    const size_t mtp_vocab =
        geometry.mtp_terminal_logits_layout ==
                MTPTerminalLogitsLayout::FullVocabularyPerParticipant
            ? static_cast<size_t>(std::max(0, profile.vocab_size))
            : local_vocab;
    size_t mtp_target_columns = checkedMultiply(
        3u, d_model, "MTP hidden/projection/output columns");
    mtp_target_columns = checkedAdd(
        mtp_target_columns, attention_output, "MTP attention output columns");
    mtp_target_columns = checkedAdd(
        mtp_target_columns,
        checkedMultiply(2u, local_d_ff, "MTP FFN projection columns"),
        "MTP FFN projection columns");
    mtp_target_columns = checkedAdd(
        mtp_target_columns, mtp_vocab, "MTP terminal logits columns");
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(
            target_rows, mtp_target_columns, "MTP target-row arena"),
        "MTP target-row arena");

    /* Stable selected-row input and the exact retained terminal output bank. */
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(1u, d_model, "LM-head selected row"),
        "LM-head selected row");
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(
            target_rows, d_model, "LM-head verifier rows"),
        "LM-head verifier rows");
    const auto mtp_gather = MTPTerminalGatherGeometry::resolve(
        geometry.mtp_terminal_logits_layout, target_rows,
        static_cast<size_t>(profile.vocab_size));
    bytes = checkedAdd(bytes,
        fp32RowBankBytes(mtp_gather.rows(), mtp_gather.columns(), "MTP gathered logits"),
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

        size_t moe_columns = checkedMultiply(
            2u, top_k, "MoE router index/weight columns");
        moe_columns = checkedAdd(
            moe_columns,
            checkedMultiply(2u, d_model, "MoE compact output columns"),
            "MoE compact output columns");
        if (geometry.routed_compute_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
        {
            if (!device.is_gpu())
                throw std::invalid_argument("MoE projection arena requires a native GPU domain");
            // Exactly the schema's replacement owner set, not an extra reserve
            // beside the complete-expert route tensor that no longer exists.
            bytes = checkedAdd(bytes,
                MoEProjectionArenaGeometry::resolve(profile, geometry.total_shards).bytes(mtp_prefill_rows),
                "MoE projection transaction arena");
        }
        else
        {
            const size_t canonical_slots = checkedAdd(top_k, canonical_participants, "MoE canonical publication slots");
            moe_columns = checkedAdd(moe_columns,
                checkedMultiply(canonical_slots, d_model, "MoE canonical route-contribution columns"),
                "MoE canonical route-contribution columns");
        }
        moe_columns = checkedAdd(
            moe_columns,
            checkedMultiply(
                2u,
                max_expert_intermediate,
                "MoE gate/up scratch columns"),
            "MoE gate/up scratch columns");
        bytes = checkedAdd(
            bytes,
            fp32RowBankBytes(
                mtp_prefill_rows, moe_columns, "MoE graph arena"),
            "MoE graph arena");
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
