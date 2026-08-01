/**
 * @file DeviceMoERebalanceABI.h
 * @brief Minimal cross-backend ABI constants for device-owned MoE rebalance.
 *
 * CUDA and ROCm kernel translation units must validate their private device
 * views against the host controller without including the host tensor, SIMD,
 * and MPI dependency graph. This header is the deliberately tiny common
 * boundary. Keep policy decisions in DeviceMoERebalancePolicyShared.h and keep
 * concrete host records in DeviceMoERebalanceController.h.
 */

#pragma once

#include <cstdint>

namespace llaminar2::moe_rebalance_abi
{
    /**
     * @brief Version of the fixed host/CUDA/ROCm publication records.
     *
     * Any field-layout change to a cross-backend record must increment this
     * value and update the corresponding byte-size assertion below.
     */
    inline constexpr uint32_t kVersion = 9u;

    /**
     * @brief Exact byte size of DeviceMoERebalanceConfig and device views.
     *
     * Active occupancy and directory addressability are deliberately separate
     * fields.  A transfer arrival may promote any staging-origin slot into a
     * durable resident slot, so every backend must preserve both capacities in
     * exactly the same ABI order.
     */
    inline constexpr uint32_t kConfigBytes = 140u;

    /**
     * @brief Exact byte size of DeviceMoERebalanceStatus and device views.
     *
     * The host type and both accelerator views statically assert this value.
     * A diagnostic field can therefore never shift only one backend silently.
     */
    inline constexpr uint32_t kStatusBytes = 648u;

    /**
     * @brief Identify payload movement planned from the current routed batch.
     *
     * Prefix-runtime rehydration and steady-state decode maintenance share the
     * same graph-owned transfer machinery. This semantic tag survives command
     * projection so the device apply kernel can publish unambiguous coverage
     * evidence without consulting the host or inferring intent from mutable
     * placement state.
     */
    inline constexpr uint32_t kPlanFlagCurrentBatchLLEP = 1u << 1;
} // namespace llaminar2::moe_rebalance_abi
