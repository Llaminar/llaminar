/**
 * @file TPContextFactory.h
 * @brief Factory for creating ITPContext instances (LOCAL or GLOBAL)
 *
 * Provides centralized creation of tensor parallelism contexts based on
 * configuration from RankExecutionPlan, TPDomainParticipation, or explicit parameters.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "ITPContext.h"
#include "ILocalTPContext.h"
#include "IGlobalTPContext.h"
#include "../backends/GlobalDeviceAddress.h"
#include "../config/OrchestrationConfig.h"
#include <memory>
#include <vector>
#include <mpi.h>

namespace llaminar2 {

// Forward declarations
struct RankExecutionPlan;
struct TPDomainParticipation;

/**
 * @brief Factory for creating LOCAL and GLOBAL tensor parallelism contexts
 *
 * Usage patterns:
 * 1. From RankExecutionPlan: TPContextFactory::create(plan, comm)
 * 2. Explicit local: TPContextFactory::createLocal(devices, weights, backend)
 * 3. Explicit global: TPContextFactory::createGlobal(comm, domain_id, color, key)
 * 4. From domain participation: TPContextFactory::createFromDomain(domain, comm)
 *
 * The factory automatically determines LOCAL vs GLOBAL based on configuration.
 */
class TPContextFactory {
public:
    // =========================================================================
    // Main Factory Methods
    // =========================================================================
    
    /**
     * @brief Create appropriate TP context based on RankExecutionPlan
     *
     * Decision logic:
     * - If plan.usesGlobalTP() → create GlobalTPContext
     * - If plan.usesLocalTP() → create LocalTPContext
     * - Otherwise → return nullptr (no TP configured)
     *
     * Note: If both local and global TP are configured, creates GlobalTPContext
     * (global TP takes precedence as it encompasses local).
     *
     * @param plan Execution plan for this rank
     * @param base_comm Base MPI communicator (for global TP comm creation)
     * @return ITPContext* (LocalTPContext or GlobalTPContext), or nullptr if no TP
     */
    static std::unique_ptr<ITPContext> create(
        const RankExecutionPlan& plan,
        MPI_Comm base_comm = MPI_COMM_WORLD);
    
    /**
     * @brief Create from TPDomainParticipation
     *
     * Analyzes domain to determine if it's local (all devices same rank)
     * or global (devices span multiple ranks).
     *
     * @param domain Domain participation info
     * @param base_comm Base MPI communicator
     * @return Appropriate ITPContext for the domain
     */
    static std::unique_ptr<ITPContext> createFromDomain(
        const TPDomainParticipation& domain,
        MPI_Comm base_comm = MPI_COMM_WORLD);

    // =========================================================================
    // Explicit Local TP Creation
    // =========================================================================
    
    /**
     * @brief Create LocalTPContext with explicit parameters
     *
     * @param devices Devices participating in LOCAL TP
     * @param weights Work distribution weights (empty for equal)
     * @param backend Backend type (AUTO to detect)
     * @return LocalTPContext instance
     */
    static std::unique_ptr<ILocalTPContext> createLocal(
        const std::vector<GlobalDeviceAddress>& devices,
        const std::vector<float>& weights = {},
        CollectiveBackendType backend = CollectiveBackendType::AUTO);
    
    /**
     * @brief Create LocalTPContext from plan's local_tp fields
     *
     * @param plan Execution plan with local_tp_devices, local_tp_weights, local_tp_backend
     * @return LocalTPContext instance, or nullptr if no local TP configured
     */
    static std::unique_ptr<ILocalTPContext> createLocalFromPlan(
        const RankExecutionPlan& plan);

    // =========================================================================
    // Explicit Global TP Creation
    // =========================================================================
    
    /**
     * @brief Create GlobalTPContext with MPI_Comm_split parameters
     *
     * @param base_comm Base MPI communicator
     * @param domain_id Domain identifier
     * @param color MPI_Comm_split color (ranks with same color form domain)
     * @param key MPI_Comm_split key (determines ordering)
     * @return GlobalTPContext instance
     */
    static std::unique_ptr<IGlobalTPContext> createGlobal(
        MPI_Comm base_comm,
        int domain_id,
        int color,
        int key);
    
    /**
     * @brief Create GlobalTPContext from plan's global_tp fields
     *
     * Uses plan.global_tp_domain_id, plan.global_tp_rank_in_domain to configure.
     *
     * @param plan Execution plan with global_tp fields
     * @param base_comm Base MPI communicator for comm_split
     * @return GlobalTPContext instance, or nullptr if no global TP configured
     */
    static std::unique_ptr<IGlobalTPContext> createGlobalFromPlan(
        const RankExecutionPlan& plan,
        MPI_Comm base_comm = MPI_COMM_WORLD);
    
private:
    // Disable instantiation - static factory only
    TPContextFactory() = delete;
};

} // namespace llaminar2
