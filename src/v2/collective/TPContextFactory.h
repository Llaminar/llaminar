/**
 * @file TPContextFactory.h
 * @brief Factory for creating ITPContext instances (LOCAL or GLOBAL)
 *
 * Provides centralized creation of tensor parallelism contexts based on
 * configuration from RankExecutionPlan, TPDomainParticipation, or explicit parameters.
 * Resolved execution scope is mandatory when projecting a domain. Device
 * hostnames and collective backend names cannot establish process membership.
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
#include <optional>
#include <mpi.h>

namespace llaminar2
{

    // Forward declarations
    struct RankExecutionPlan;
    struct TPDomainParticipation;

    /**
     * @brief Factory for creating LOCAL and GLOBAL tensor parallelism contexts
     *
     * Usage patterns:
     * 1. From RankExecutionPlan: TPContextFactory::create(plan, comm)
     * 2. Explicit local: TPContextFactory::createLocal(devices, weights, backend)
     * 3. Explicit global: createGlobal(comm, domain_id, color, key, resolved_cpu)
     * 4. From resolved participation: TPContextFactory::createFromDomain(domain, scope, comm)
     *
     * The factory automatically determines LOCAL vs GLOBAL based on configuration.
     */
    class TPContextFactory
    {
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
            const RankExecutionPlan &plan,
            MPI_Comm base_comm);

        /**
         * @brief Create from TPDomainParticipation
         *
         * Consumes the canonical resolved scope: rank-local execution creates
         * LocalTPContext; node-local/cross-node rank execution creates
         * GlobalTPContext. Physical node membership inside that communicator
         * still comes from MPI inventory, never the selected transport name.
         *
         * @param domain Domain participation info
         * @param resolved_scope Canonical resolved scope; AUTO is not executable.
         * @param base_comm Base MPI communicator
         * @return Appropriate ITPContext for the domain
         * @throws std::invalid_argument for unresolved or inconsistent scope.
         */
        static std::unique_ptr<ITPContext> createFromDomain(
            const TPDomainParticipation &domain,
            ExecutionDomainScope resolved_scope,
            MPI_Comm base_comm);

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
            const std::vector<GlobalDeviceAddress> &devices,
            const std::vector<float> &weights = {},
            CollectiveBackendType backend = CollectiveBackendType::AUTO);

        /**
         * @brief Create LocalTPContext from plan's local_tp fields
         *
         * @param plan Execution plan with local_tp_devices, local_tp_weights, local_tp_backend
         * @return LocalTPContext instance, or nullptr if no local TP configured
         */
        static std::unique_ptr<ILocalTPContext> createLocalFromPlan(
            const RankExecutionPlan &plan);

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
         * @param local_device Exact resolved CPU endpoint, absent only when excluded.
         * @return GlobalTPContext instance
         */
        static std::unique_ptr<IGlobalTPContext> createGlobal(
            MPI_Comm base_comm,
            int domain_id,
            int color,
            int key,
            std::optional<GlobalDeviceAddress> local_device,
            const std::string &hostfile_path = "",
            CollectiveBackendType backend = CollectiveBackendType::UPI);

        /**
         * @brief Create GlobalTPContext from plan's global_tp fields
         *
         * Uses plan.global_tp_domain_id, plan.global_tp_rank_in_domain to configure.
         *
         * @param plan Execution plan with global_tp fields
         * @param base_comm Base MPI communicator for comm_split
         * @param hostfile_path Optional MPI hostfile for node detection ordering
         * @return GlobalTPContext instance, or nullptr if no global TP configured
         */
        static std::unique_ptr<IGlobalTPContext> createGlobalFromPlan(
            const RankExecutionPlan &plan,
            MPI_Comm base_comm,
            const std::string &hostfile_path = "");

    private:
        // Disable instantiation - static factory only
        TPContextFactory() = delete;
    };

} // namespace llaminar2
