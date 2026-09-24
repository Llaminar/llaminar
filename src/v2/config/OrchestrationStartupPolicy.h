/**
 * @file OrchestrationStartupPolicy.h
 * @brief Publish explicit configuration into the process startup-policy view.
 *
 * The typed config preserves plan/apply intent; DebugEnv is the existing
 * kernel/capture consumer view. This adapter runs only before runtime admission
 * and keeps direct launches and newly exec'd MPI children equivalent. It does
 * not change live graph identities or introduce another memory authority.
 */
#pragma once

namespace llaminar2
{
    struct OrchestrationConfig;
    /**
     * @brief Publish explicit prefill and deterministic startup settings once.
     * @param config Fully parsed configuration, before any model admission.
     * @throws std::invalid_argument for invalid explicit prefill capacity.
     * @throws std::runtime_error when the process environment cannot be updated.
     *
     * Omitted settings leave inherited diagnostic policy untouched. Reapplication
     * is idempotent; this must never be called during active model execution.
     */
    void publishOrchestrationStartupPolicy(const OrchestrationConfig &config);
}
