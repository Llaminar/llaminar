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
        geometry.local_n_kv_heads < 0 || geometry.total_shards <= 0 ||
        geometry.mtp_target_query_rows <= 0)
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
    const size_t local_vocab = shardColumns(
        static_cast<size_t>(std::max(0, profile.vocab_size)),
        geometry.total_shards);

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

    const size_t full_attention_q = shardColumns(
        projectionOutputRows(
            profile,
            ".attn_q.weight",
            inventory_first_layer,
            inventory_last_layer),
        geometry.total_shards);
    const size_t full_attention_gate = shardColumns(
        projectionOutputRows(
            profile,
            ".attn_gate.weight",
            inventory_first_layer,
            inventory_last_layer),
        geometry.total_shards);
    const size_t gdn_qkv = shardColumns(
        projectionOutputRows(
            profile,
            ".attn_qkv.weight",
            inventory_first_layer,
            inventory_last_layer),
        geometry.total_shards);
    const size_t gdn_z = shardColumns(
        projectionInputRows(
            profile,
            ".ssm_out.weight",
            inventory_first_layer,
            inventory_last_layer),
        geometry.total_shards);
    const size_t gdn_alpha = shardColumns(
        projectionOutputRows(
            profile,
            ".ssm_alpha.weight",
            inventory_first_layer,
            inventory_last_layer),
        geometry.total_shards);
    const size_t gdn_beta = shardColumns(
        projectionOutputRows(
            profile,
            ".ssm_beta.weight",
            inventory_first_layer,
            inventory_last_layer),
        geometry.total_shards);

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

    /* Stable selected-row input plus the no-global-gather 1x1 placeholder. */
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(1u, d_model, "LM-head selected row"),
        "LM-head selected row");
    bytes = checkedAdd(
        bytes,
        fp32RowBankBytes(
            target_rows, d_model, "LM-head verifier rows"),
        "LM-head verifier rows");
    bytes = checkedAdd(bytes, sizeof(float), "MTP gather placeholder");

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
        const size_t canonical_slots = checkedAdd(
            top_k,
            canonical_participants,
            "MoE canonical publication slots");
        moe_columns = checkedAdd(
            moe_columns,
            checkedMultiply(
                canonical_slots,
                d_model,
                "MoE canonical route-contribution columns"),
            "MoE canonical route-contribution columns");
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
    return bytes;
}

} // namespace llaminar2
