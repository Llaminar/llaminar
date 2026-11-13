/**
 * @file DeviceMoECommittedMaintenanceScope.h
 * @brief Cold policy projection from original runtime capacity to committed main routing evidence.
 *
 * Model-owned tables can also retain speculative NextN placement descriptors.
 * Post-commit maintenance observes only the original committed main interval.
 */
#pragma once
#include "DeviceMoERebalanceController.h"
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Scope post-commit maintenance to layers with committed main-model histograms.
     * @param config Original policy, including complete materialized runtime table capacity.
     * @param first_main_layer First main transformer layer in runtime-table storage coordinates.
     * @param main_layer_count Actual main transformer interval owned by this stage.
     * @return Policy retaining table capacity and targeting only the committed main interval.
     * @throws std::invalid_argument If the main interval is empty or outside the original table.
     *
     * NextN source layers retain placement descriptors in the same table, but
     * speculative predictor execution does not author committed routing counts.
     * Dedicated maintenance runs after commit and can consume the first main
     * layer, unlike an inline producer that executes before that layer's router.
     */
    inline DeviceMoERebalanceConfig deviceMoECommittedMainLayerScope(
        DeviceMoERebalanceConfig config, uint32_t first_main_layer, uint32_t main_layer_count)
    {
        if (!main_layer_count || first_main_layer >= config.num_layers ||
            main_layer_count > config.num_layers - first_main_layer)
            throw std::invalid_argument("Committed MoE main layer interval is outside its original runtime table");
        config.layer_window_start = first_main_layer;
        config.layer_window_count = main_layer_count;
        return config;
    }

} // namespace llaminar2
