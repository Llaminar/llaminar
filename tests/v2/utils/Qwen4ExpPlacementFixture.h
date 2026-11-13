/**
 * @file Qwen4ExpPlacementFixture.h
 * @brief Canonical selector admission for device-free graph declaration fixtures.
 *
 * Test budgets are explicit metadata; this helper never allocates model or GPU
 * storage. Real materialization fixtures admit their own exact physical owners.
 */
#pragma once
#include "models/GraphTypes.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include <array>
namespace llaminar2::test
{
/**
 * @brief Use the real selector to authorize a metadata-only test consumer.
 * @param model Exact graph metadata; only an owning PLE interval gets placement.
 * @param requested Explicit requested tier, independent of the concrete result.
 * No model file, allocator, or accelerator participates in this unit fixture.
 */
inline void admitQwen4ExpTestTable(GraphConfig &model,
    ImmutableRowStoragePreference requested = ImmutableRowStoragePreference::RAM)
{
    const auto geometry = Qwen4ExpMemoryGeometry::fromConfig(*model.flash_next);
    const int ple = static_cast<int>(geometry.ple_decoder_layer);
    if (ple < model.pp_layer_offset || ple >= model.pp_layer_offset + model.n_layers) return;
    constexpr std::size_t capacity = std::size_t{1} << 40;
    const PhysicalMemoryResource host{0, DeviceId::cpu(), capacity, capacity};
    const std::array consumers{PhysicalMemoryResource{0, model.default_device, capacity, capacity}};
    PhysicalMemoryPlanBuilder base;
    base.add(consumers.front(), PhysicalMemoryOwner::PrimaryModelWeights, 1u);
    const auto extent = ImmutableRowTableExtent::fromNative(
        static_cast<GGUFTensorType>(geometry.ple_table_gguf_type),
        geometry.ple_table_rows, geometry.ple_table_row_bytes);
    model.ngram_table_placement = ImmutableRowTablePlacementPlanner::select(
        ImmutableRowTablePolicy{requested},
        ImmutableRowTablePlacementInput::forConsumers(extent, host, consumers),
        base.build(), {.vram=PhysicalMemoryPlan{}}).placement;
}
}
