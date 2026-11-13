/**
 * @file Qwen35MoEParityEconomics.h
 * @brief Canonical short-workload expert movement policy without snapshot dependencies.
 *
 * Model-free native lifecycle tests and real-model parity consume this same
 * policy. Keeping the declaration separate avoids importing numerical snapshot
 * readers merely to select the existing maintenance and payoff parameters.
 */
#pragma once
#include "execution/config/RuntimeConfig.h"

namespace llaminar2::test::parity::qwen35moe {
/** @return Short-test economics that still require measured positive payoff. */
inline MoERebalanceRuntimeConfig qwen35MoEDynamicParityEconomics()
{
    MoERebalanceRuntimeConfig config;
    config.mode = MoERebalanceRuntimeMode::Dynamic;
    config.window_size = 1;
    config.max_window_size = 1;
    config.window_growth_factor = 1.0f;
    config.dynamic_imbalance_threshold_per_mille = 1000;
    config.dynamic_min_improvement_per_mille = 0;
    config.dynamic_max_swaps_per_layer = 20;
    config.dynamic_max_plan_entries_per_wave = 20;
    config.dynamic_min_window_activations = 0;
    config.device_min_load_spread_improvement = 0;
    config.device_min_load_spread_improvement_divisor = 0;
    config.device_min_wave_spread_improvement_per_payload_slot = 0;
    config.device_min_foreign_rows_per_critical_path_payload_slot = 0;
    config.device_min_router_spread_improvement_per_payload_slot = 0;
    config.device_max_post_wave_load_spread_per_mille = 1000;
    config.device_maintenance_slack_tokens = 0;
    config.device_min_maintenance_period_tokens = 1;
    config.device_initial_maintenance_period_tokens = 1;
    config.migration_payoff_horizon_tokens = 65'536;
    config.release_raw_expert_weights = false;
    return config;
}
} // namespace llaminar2::test::parity::qwen35moe
