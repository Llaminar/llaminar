/**
 * @file MoEProjectionArenaGeometry.cpp
 * @brief Metadata-only sizing and declaration of distributed MoE row banks.
 *
 * A quantized down projection already consumes block-Q8 with FP32 scales;
 * floating down consumes FP32. Preserve that distinction while computing the
 * maximum live row bank over all main/sidecar layers. No bytes are allocated
 * here: these definitions feed the canonical arena and physical memory ledger.
 */
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "execution/moe/MoEGroupedIntermediateExchangeABI.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "loaders/WeightIdentity.h"
#include "planning/ModelMemoryProfile.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
namespace
{
    /** @brief Reject overflow before an arena size can reach admission or allocation. */
    std::size_t multiply(std::size_t a, std::size_t b)
    {
        if (a && b > std::numeric_limits<std::size_t>::max() / a)
            throw std::overflow_error("MoE projection arena size overflow");
        return a * b;
    }
}

MoEProjectionArenaGeometry::MoEProjectionArenaGeometry(
    std::size_t packet_words, std::size_t top_k, std::size_t columns, std::size_t participants)
    : banks_{{
        {BufferId::MOE_PROJECTION_LOCAL_PACKET, "moe_projection_local_packet", packet_words},
        {BufferId::MOE_PROJECTION_GATHERED_PACKETS, "moe_projection_gathered_packets", multiply(packet_words, participants)},
        {BufferId::MOE_PROJECTION_ROUTE_COLUMNS, "moe_projection_route_columns", multiply(top_k, columns / participants)},
        {BufferId::MOE_PROJECTION_LOCAL_COLUMNS, "moe_projection_local_columns", columns / participants},
        {BufferId::MOE_PROJECTION_GATHERED_COLUMNS, "moe_projection_gathered_columns", columns}}}
{}

MoEProjectionArenaGeometry MoEProjectionArenaGeometry::resolve(
    const ModelMemoryProfile &profile, int participants)
{
    if (participants < 2 || profile.d_model <= 0 || profile.d_model % participants ||
        profile.n_layers <= 0 || profile.expert_count <= 0 ||
        profile.expert_used_count <= 0 || profile.expert_used_count > profile.expert_count)
        throw std::invalid_argument("MoE projection arena requires complete, divisible native-domain geometry");

    std::vector<bool> seen(static_cast<std::size_t>(profile.n_layers), false);
    std::size_t packet_words = 0;
    for (const auto &tensor : profile.tensors)
    {
        if (inferWeightRole(tensor.name) != WeightRole::MoEExpertDown) continue;
        if (tensor.layer_index < 0 || tensor.layer_index >= profile.n_layers || seen[tensor.layer_index] ||
            tensor.K == 0 || tensor.K > std::numeric_limits<std::uint32_t>::max() ||
            tensor.elements != multiply(multiply(profile.d_model, tensor.K), profile.expert_count))
            throw std::invalid_argument("MoE projection arena has missing/duplicate or malformed down geometry");

        const bool floating = tensor.quant_type == "F16" || tensor.quant_type == "FP16" ||
            tensor.quant_type == "BF16" || tensor.quant_type == "F32" || tensor.quant_type == "FP32";
        if (!floating && !native_vnni_formats::forQuantType(tensor.quant_type))
            throw std::invalid_argument("MoE projection arena has an unsupported down format: " + tensor.quant_type);
        const MoEGroupedIntermediateLayout row{
            floating ? MoEGroupedIntermediateEncoding::FP32 : MoEGroupedIntermediateEncoding::BlockQ8FP32Scales,
            static_cast<std::uint32_t>(tensor.K), static_cast<std::uint32_t>(profile.expert_used_count),
            static_cast<std::uint32_t>(participants)};
        if (!row.compactValid()) throw std::invalid_argument("MoE projection intermediate row is not representable");
        // Each layer reuses this allocation. A floating sidecar or wider layer
        // can set the maximum, but must not inflate all-quantized models to FP32.
        packet_words = std::max(packet_words, row.compactCapacityBytes() / sizeof(std::uint32_t));
        seen[tensor.layer_index] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::invalid_argument("MoE projection arena requires every main and MTP down projection");
    return MoEProjectionArenaGeometry(packet_words, profile.expert_used_count, profile.d_model, participants);
}

std::size_t MoEProjectionArenaGeometry::bytes(std::size_t rows) const
{
    if (!rows) throw std::invalid_argument("MoE projection arena requires positive row capacity");
    std::size_t total = 0;
    for (const auto &bank : banks_)
    {
        const auto bytes = multiply(multiply(rows, bank.words_per_row), sizeof(std::uint32_t));
        if (bytes > std::numeric_limits<std::size_t>::max() - total)
            throw std::overflow_error("MoE projection arena total overflow");
        total += bytes;
    }
    return total;
}

std::size_t MoEProjectionArenaGeometry::packetCapacityBytes(std::size_t rows) const
{
    if (!rows) throw std::invalid_argument("MoE projection packet requires positive row capacity");
    return multiply(multiply(rows, banks_[0].words_per_row), sizeof(std::uint32_t));
}

void MoEProjectionArenaGeometry::replaceWholeExpertSchema(GraphSchema &schema) const
{
    const auto is_canonical = [](const BufferSpec &spec) { return spec.name == "moe_canonical_route_contributions"; };
    if (std::count_if(schema.layer_buffers.begin(), schema.layer_buffers.end(), is_canonical) != 1)
        throw std::invalid_argument("MoE projection schema must replace exactly one whole-expert publication bank");
    for (const auto &bank : banks_)
        if (std::any_of(schema.layer_buffers.begin(), schema.layer_buffers.end(),
                [&](const BufferSpec &spec) { return spec.name == bank.name; }))
            throw std::invalid_argument("MoE projection arena already declared");
    std::erase_if(schema.layer_buffers, is_canonical);
    for (const auto &bank : banks_)
    {
        // FP32 storage is also an exact 32-bit packet carrier. Kernels treat
        // packet payloads as opaque words, never as converted float values.
        const std::string name(bank.name);
        schema.layer_buffers.push_back({name, {"moe_activation_rows", name + "_words"}, "fp32",
            BufferSemantic::Scratch, name, 10, "Native MoE projection transaction row bank"});
    }
}

void MoEProjectionArenaGeometry::bindResolver(GraphResolverConfig &resolver) const
{
    for (const auto &bank : banks_)
    {
        const std::string name(bank.name);
        resolver.buffer_name_to_id[name] = bank.id;
        resolver.custom_formulas[name + "_words"] = bank.words_per_row;
    }
}
} // namespace llaminar2
