/**
 * @file MoEIntermediateShardArenaGeometry.cpp
 * @brief Compose canonical native source cuts with original shared TP arena widths.
 *
 * The source inventory supplies N/K/E and native codebook groups. The existing
 * intermediate partitioner supplies routed cuts; existing TP d_ff fields
 * supply shared cuts. This file neither reconstructs placement nor subtracts
 * capacity. Every main and predictor layer contributes to the retained maximum.
 */
#include "MoEIntermediateShardArenaGeometry.h"
#include "MoEExpertProjectionOwnership.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "loaders/WeightIdentity.h"
#include "planning/ModelMemoryProfile.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/BlockStructures.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>
namespace llaminar2
{
MoEIntermediateShardArenaGeometry MoEIntermediateShardArenaGeometry::resolve(const ModelMemoryProfile &source,
    int participants, int participant, const std::optional<DeviceShardingAssignment> &shared_assignment,
    int first_layer, int last_layer)
{
    if (participants <= 0 || participant < 0 || participant >= participants || source.d_model <= 0 ||
        source.expert_count <= 0 || source.expert_used_count <= 0 || source.expert_used_count > source.expert_count ||
        source.totalLayerCount() <= 0 || (participants > 1 && !shared_assignment))
        throw std::invalid_argument("Intermediate-shard arena requires original positive source/cohort geometry");
    const auto layers = source.totalLayerCount();
    if (last_layer < 0) last_layer = layers - 1;
    if (first_layer < 0 || first_layer > last_layer || last_layer >= layers)
        throw std::invalid_argument("Intermediate-shard arena has a foreign retained source-layer interval");
    std::vector<std::array<bool, 3>> routed_seen(layers), shared_seen(layers);
    std::vector<int> original_routed_f(layers, 0);
    int routed_max = 0, shared_source = 0;
    for (const auto &tensor : source.tensors)
    {
        if (tensor.layer_index < first_layer || tensor.layer_index > last_layer) continue;
        const auto role = inferWeightRole(tensor.name);
        const bool routed = isRoutedExpertRole(role);
        const bool shared = role == WeightRole::SharedExpertGate || role == WeightRole::SharedExpertUp || role == WeightRole::SharedExpertDown;
        if (!routed && !shared) continue;
        if (tensor.layer_index < 0 || tensor.layer_index >= layers || !tensor.K || !tensor.elements)
            throw std::invalid_argument("Intermediate-shard arena has foreign or empty FFN source metadata");
        const bool down = role == WeightRole::MoEExpertDown || role == WeightRole::SharedExpertDown;
        const int axis = down ? 2 : role == WeightRole::MoEExpertGate || role == WeightRole::SharedExpertGate ? 0 : 1;
        auto &seen = routed ? routed_seen[tensor.layer_index] : shared_seen[tensor.layer_index];
        if (seen[axis]) throw std::invalid_argument("Intermediate-shard arena repeats an original projection");
        seen[axis] = true;
        const auto population = routed ? std::size_t(source.expert_count) : 1u;
        const auto denominator = population * std::size_t(source.d_model);
        if (tensor.elements % denominator || tensor.elements / denominator > std::size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Intermediate-shard arena has unrepresentable original F geometry");
        const int f = int(tensor.elements / denominator);
        if (f <= 0 || tensor.K != std::size_t(down ? f : source.d_model))
            throw std::invalid_argument("Intermediate-shard arena changed original N/K/E axes");
        if (shared)
        {
            if (shared_source && shared_source != f)
                throw std::invalid_argument("Intermediate-shard shared source F varies outside its original dense TP assignment");
            shared_source = f;
            if (down && shared_assignment)
            {
                const bool floating = tensor.quant_type == "F32" || tensor.quant_type == "FP32" ||
                    tensor.quant_type == "F16" || tensor.quant_type == "FP16" || tensor.quant_type == "BF16";
                const auto *native = floating ? nullptr : native_quantised_formats::forQuantType(tensor.quant_type);
                if (!floating && !native)
                    throw std::invalid_argument("Intermediate-shard arena has unsupported original shared down codebook");
                const int group = floating ? 1 : native->is_superblock ? Q4_KBlock::BLOCK_SIZE : Q4_0Block::BLOCK_SIZE;
                if (shared_assignment->d_ff_start % group || shared_assignment->d_ff_count % group)
                    throw std::invalid_argument("Intermediate-shard original shared TP cut splits a native down codebook group");
            }
        }
        else
        {
            auto &original_f = original_routed_f[tensor.layer_index];
            if (original_f && original_f != f)
                throw std::invalid_argument("Intermediate-shard routed triplet disagrees on its original F axis");
            original_f = f;
            if (!down) continue;
            const bool floating = tensor.quant_type == "F32" || tensor.quant_type == "FP32" ||
                tensor.quant_type == "F16" || tensor.quant_type == "FP16" || tensor.quant_type == "BF16";
            const auto *native = floating ? nullptr : native_quantised_formats::forQuantType(tensor.quant_type);
            if (!floating && !native) throw std::invalid_argument("Intermediate-shard arena has an unsupported original down codebook");
            const int group = floating ? 1 : native->is_superblock ? Q4_KBlock::BLOCK_SIZE : Q4_0Block::BLOCK_SIZE;
            const auto intervals = MoEExpertProjectionOwnership::nativeIntermediateIntervals(f, participants, group, tensor.layer_index);
            routed_max = std::max(routed_max, intervals[participant].columns);
        }
    }
    const auto complete = [&](const auto &seen) {
        return std::all_of(seen.begin() + first_layer, seen.begin() + last_layer + 1, [](const auto &layer) { return layer[0] && layer[1] && layer[2]; });
    };
    if (!complete(routed_seen) || !complete(shared_seen) || !routed_max || !shared_source)
        throw std::invalid_argument("Intermediate-shard arena requires complete main/predictor routed and shared source triplets");
    int shared_local = shared_source;
    if (shared_assignment)
    {
        const auto &a = *shared_assignment;
        if (a.local_rank != participant || !a.device.is_gpu() || a.d_ff_start < 0 || a.d_ff_count <= 0 ||
            a.d_ff_start > shared_source || a.d_ff_count > shared_source - a.d_ff_start)
            throw std::invalid_argument("Intermediate-shard arena has a foreign original shared source assignment");
        shared_local = a.d_ff_count;
    }
    return {routed_max, shared_local, source.expert_used_count};
}
int MoEIntermediateShardArenaGeometry::scratchIntermediate() const noexcept { return std::max(routed_, shared_); }
void MoEIntermediateShardArenaGeometry::bindResolver(GraphResolverConfig &resolver) const
{
    resolver.custom_formulas["moe_top_k"] = std::size_t(slots_);
    resolver.custom_formulas["moe_canonical_publication_slots"] = std::size_t(slots_);
    resolver.custom_formulas["moe_expert_intermediate"] = std::size_t(routed_);
    resolver.custom_formulas["moe_ffn_intermediate_max"] = std::size_t(scratchIntermediate());
}
void MoEIntermediateShardArenaGeometry::bindSchema(GraphSchema &schema) const
{
    for (const auto &name : {"moe_combined_output", "moe_shared_expert_output", "moe_canonical_route_contributions"})
    {
        auto found = std::find_if(schema.layer_buffers.begin(), schema.layer_buffers.end(),
            [&](const auto &entry) { return entry.name == name; });
        if (found == schema.layer_buffers.end()) throw std::invalid_argument("Intermediate-shard arena is missing its declared local partial bank");
        // Producers coexist through gate/add; their canonical roles cannot alias.
        found->alias_group = name;
    }
}
}
