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
    struct ModelMemoryProfile;

    /**
     * @brief Seal MTP activation from authored intent and the canonical GGUF directory.
     * @param config Startup configuration; activation is concrete on return.
     * @param model Source-bound metadata with the learned-head tensor inventory.
     * @throws std::invalid_argument for missing required or incomplete learned heads.
     *
     * Explicit disablement remains a first-class configuration on MTP GGUFs.
     * Disabled execution with positive retained capacity still authenticates heads.
     */
    void resolveMTPStartupPolicy(OrchestrationConfig &config, const ModelMemoryProfile &model);
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
