#include "planning/ActivationMemoryEstimator.h"

#include <algorithm>
#include <string_view>

namespace llaminar2
{

namespace
{

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

    size_t B = static_cast<size_t>(batch_size);
    size_t S = static_cast<size_t>(max_seq_len);
    size_t D = static_cast<size_t>(d_model);
    size_t F = static_cast<size_t>(d_ff);
    size_t H = static_cast<size_t>(n_heads);
    size_t HK = static_cast<size_t>(n_kv_heads);
    size_t HD = static_cast<size_t>(head_dim);
    size_t V = static_cast<size_t>(vocab_size);
    (void)device;

    // All sizes in bytes (FP32 = 4 bytes per element unless noted)
    constexpr size_t FP32 = 4;

    const size_t Q = H * HD;
    const size_t KV = HK * HD;

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
    const size_t resident_row_elements =
        5 * D + 2 * Q + 2 * KV + 2 * F + S;
    const size_t resident_buffers = B * S * resident_row_elements * FP32;

    // Ordinary prefill/decode computes only the selected terminal logits row.
    const size_t terminal_logits = B * V * FP32;
    return resident_buffers + terminal_logits;
}

size_t ActivationMemoryEstimator::estimate(
    const ModelMemoryProfile& profile,
    int batch_size,
    int resident_graph_rows,
    int local_d_ff,
    int local_n_heads,
    int local_n_kv_heads,
    int first_layer,
    int last_layer,
    int total_shards,
    DeviceId device)
{
    size_t bytes = estimate(
        batch_size,
        resident_graph_rows,
        profile.d_model,
        local_d_ff,
        local_n_heads,
        local_n_kv_heads,
        profile.head_dim,
        profile.vocab_size,
        device);
    if (bytes == 0)
        return 0;

    /*
     * Qwen3.5/3.6 hybrid layers add physical BufferIds beyond the common
     * transformer schema. Infer their exact row widths from projection tensor
     * geometry so planning follows the model file rather than a model-name
     * heuristic.
     */
    if (!hasHybridRecurrentLayer(profile, first_layer, last_layer))
        return bytes;

    const size_t full_attention_q = shardColumns(
        projectionOutputRows(profile, ".attn_q.weight", first_layer, last_layer),
        total_shards);
    const size_t full_attention_gate = shardColumns(
        projectionOutputRows(profile, ".attn_gate.weight", first_layer, last_layer),
        total_shards);
    const size_t gdn_qkv = shardColumns(
        projectionOutputRows(profile, ".attn_qkv.weight", first_layer, last_layer),
        total_shards);
    const size_t gdn_z = shardColumns(
        projectionInputRows(profile, ".ssm_out.weight", first_layer, last_layer),
        total_shards);
    const size_t gdn_alpha = shardColumns(
        projectionOutputRows(profile, ".ssm_alpha.weight", first_layer, last_layer),
        total_shards);
    const size_t gdn_beta = shardColumns(
        projectionOutputRows(profile, ".ssm_beta.weight", first_layer, last_layer),
        total_shards);

    const size_t hybrid_columns =
        full_attention_q +
        full_attention_gate +
        gdn_qkv +
        gdn_z +
        gdn_alpha +
        gdn_beta;
    bytes +=
        static_cast<size_t>(std::max(1, batch_size)) *
        static_cast<size_t>(std::max(1, resident_graph_rows)) *
        hybrid_columns *
        sizeof(float);
    return bytes;
}

} // namespace llaminar2
