#include "planning/WorkspaceMemoryEstimator.h"

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

    // Floor: 768 MB (matches WorkspaceAllocator)
    constexpr size_t FLOOR = 768ULL * 1024 * 1024;

    return std::max(FLOOR, raw);
}

size_t WorkspaceMemoryEstimator::estimate(
    const ModelMemoryProfile& profile,
    int batch_size,
    int resident_graph_rows,
    int local_d_ff,
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
        profile.vocab_size,
        device);
    if (bytes == 0)
        return 0;

    if (!hasHybridRecurrentLayer(profile, first_layer, last_layer))
        return bytes;

    const size_t gdn_qkv = shardColumns(
        projectionOutputRows(profile, ".attn_qkv.weight", first_layer, last_layer),
        total_shards);
    const size_t gdn_gate = shardColumns(
        projectionOutputRows(profile, ".attn_gate.weight", first_layer, last_layer),
        total_shards);
    if (gdn_qkv == 0 && gdn_gate == 0)
        return bytes;

    /*
     * Hybrid GDN layers add one in-place short-convolution scratch row of QKV
     * width and one three-way deinterleave scratch row of gate width.
     */
    const size_t hybrid_row_bytes =
        (gdn_qkv + 3 * gdn_gate) * sizeof(float);
    bytes +=
        static_cast<size_t>(std::max(1, batch_size)) *
        static_cast<size_t>(std::max(1, resident_graph_rows)) *
        hybrid_row_bytes;

    return bytes;
}

} // namespace llaminar2
