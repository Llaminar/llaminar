/**
 * @file GraphArenaStorageLayout.cpp
 * @brief Resolve and price the exact arena inventory consumed by graph setup.
 *
 * Filtering and name resolution occur at this common boundary so admission
 * cannot omit a root that runtime registers. Physical byte rounding remains
 * with CPUTensorStorageDeclaration and its backing allocator.
 */
#include "GraphArenaStorageLayout.h"
#include "BufferArena.h"
#include "CPUTensorStorageDeclaration.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include <limits>
#include <set>
#include <stdexcept>

namespace llaminar2
{
GraphArenaStorageLayout GraphArenaStorageLayout::resolve(const GraphSchema &schema,
    const GraphResolverConfig &resolver, GraphArenaSupplementalStorage supplemental)
{
    GraphArenaStorageLayout layout;
    std::set<BufferId> identities;
    const auto append = [&](BufferId id, BufferDescriptor descriptor)
    {
        if (id == BufferId::_COUNT || !identities.insert(id).second)
            throw std::invalid_argument("Graph arena has an unknown or duplicate root: " + descriptor.name);
        layout.entries_.push_back({id, std::move(descriptor)});
    };
    const auto append_schema = [&](const StageBufferRequirements &requirements)
    {
        for (const auto &descriptor : requirements.buffers)
        {
            auto id = BufferArena::bufferNameToId(descriptor.name);
            if (id == BufferId::_COUNT)
            {
                const auto found = resolver.buffer_name_to_id.find(descriptor.name);
                if (found != resolver.buffer_name_to_id.end()) id = found->second;
            }
            // All native attention programs use tiled softmax. These declared
            // legacy scratch roles have no consumer and are never allocated.
            if (id == BufferId::ATTN_SCORES_WORKSPACE || id == BufferId::ATTN_CONTEXT_WORKSPACE) continue;
            append(id, descriptor);
        }
    };
    append_schema(BufferAllocator::resolveLayerBuffers(schema, resolver));
    append_schema(BufferAllocator::resolveModelBuffers(schema, resolver));
    if (supplemental.full_prefill_kv_handoff)
    {
        if (resolver.seq_len <= 0 || resolver.n_kv_heads <= 0 || resolver.head_dim <= 0)
            throw std::invalid_argument("Graph arena full prefill KV handoff requires positive exact dimensions");
        const size_t heads = static_cast<size_t>(resolver.n_kv_heads);
        const size_t width = static_cast<size_t>(resolver.head_dim);
        if (heads > std::numeric_limits<size_t>::max() / width)
            throw std::overflow_error("Graph arena full prefill KV width overflows");
        for (const auto &[id, name] : {std::pair{BufferId::K_FULL_PREFILL, "K_full_prefill"},
                                      std::pair{BufferId::V_FULL_PREFILL, "V_full_prefill"}})
            append(id, {.name=name, .shape={static_cast<size_t>(resolver.seq_len), heads * width},
                .tensor_type=BufferTensorType::FP32, .device=resolver.default_device});
    }
    if ((supplemental.terminal_hidden_rows == 0) != (supplemental.terminal_hidden_columns == 0))
        throw std::invalid_argument("Graph arena terminal hidden requires both dimensions or explicit absence");
    if (supplemental.terminal_hidden_rows)
        append(BufferId::PREFIX_TERMINAL_HIDDEN, {.name="prefix_terminal_hidden",
            .shape={supplemental.terminal_hidden_rows, supplemental.terminal_hidden_columns},
            .tensor_type=BufferTensorType::FP32, .device=resolver.default_device});
    return layout;
}

size_t GraphArenaStorageLayout::cpuPhysicalBytes() const
{
    size_t bytes = 0;
    for (const auto &entry : entries_)
    {
        const size_t extent = CPUTensorStorageDeclaration::fromDescriptor(
            entry.descriptor, PhysicalMemoryOwner::ActivationArena).physicalBytes();
        if (extent > std::numeric_limits<size_t>::max() - bytes)
            throw std::overflow_error("CPU graph arena physical root sum overflows");
        bytes += extent;
    }
    return bytes;
}
}
