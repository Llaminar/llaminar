/**
 * @file ImmutableRowTableParticipantRole.h
 * @brief Cold factory participation for original immutable PLE source owners.
 *
 * Storage ownership spans PP stages while query ownership belongs to one
 * decoder layer. Decide before the factory's early return so storage-only
 * owners retain their drained ordinary-loader staging and original model/PMA.
 */
#pragma once
#include "loaders/ImmutableFixedHeadPipelineProjection.h"
#include <optional>

namespace llaminar2 {
/** @brief Distinct source preparation obligations, without introducing execution participants. */
enum class ImmutableRowTableParticipantRole { None, Consumer, FixedPipelineStorage };

/**
 * @brief Resolve the cold materialization role of one original graph participant.
 * @param ple_layer Original PLE layer, absent for predictor/non-QFN graphs.
 * @param layer_offset First decoder layer owned by this PP participant.
 * @param layer_count Number of participant-local decoder layers.
 * @param device Original graph device, never an inferred sibling device.
 * @param pipeline Original admitted fixed-range pipeline, if selected.
 * @return Fixed source ownership before query-layer filtering, or the local reader role.
 */
inline ImmutableRowTableParticipantRole immutableRowTableParticipantRole(
    std::optional<int> ple_layer, int layer_offset, int layer_count, DeviceId device,
    const ImmutableFixedHeadPipelineProjection *pipeline)
{
    if (!ple_layer) return ImmutableRowTableParticipantRole::None;
    if (pipeline) {
        for (const auto &owner : pipeline->owners())
            if (owner.allocator.device == device)
                return ImmutableRowTableParticipantRole::FixedPipelineStorage;
    }
    return *ple_layer >= layer_offset && *ple_layer - layer_offset < layer_count
        ? ImmutableRowTableParticipantRole::Consumer : ImmutableRowTableParticipantRole::None;
}
} // namespace llaminar2
