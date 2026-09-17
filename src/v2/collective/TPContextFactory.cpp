/**
 * @file TPContextFactory.cpp
 * @brief Implementation of factory for creating ITPContext instances
 *
 * Cross-rank contexts inherit the caller's admitted communicator and resolved
 * CPU endpoint. There is no implicit WORLD membership or fabricated locality.
 * Domain construction consumes the resolved execution scope, not a guess from
 * hostname spelling or the desired transport's name.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TPContextFactory.h"
#include "LocalTPContext.h"
#include "GlobalTPContext.h"
#include "../execution/mpi_orchestration/RankExecutionPlan.h"
#include "../utils/Logger.h"
#include <mpi.h>

namespace llaminar2
{

    // =============================================================================
    // Main Factory Methods
    // =============================================================================

    std::unique_ptr<ITPContext> TPContextFactory::create(
        const RankExecutionPlan &plan,
        MPI_Comm base_comm)
    {
        // Global TP takes precedence (encompasses local as a special case)
        if (plan.usesGlobalTP())
        {
            LOG_DEBUG("TPContextFactory::create - Plan uses Global TP, creating GlobalTPContext");
            return createGlobalFromPlan(plan, base_comm);
        }

        // Local TP only
        if (plan.usesLocalTP())
        {
            LOG_DEBUG("TPContextFactory::create - Plan uses Local TP, creating LocalTPContext");
            return createLocalFromPlan(plan);
        }

        // No TP configured
        LOG_DEBUG("TPContextFactory::create - No TP configured in plan (local_tp_devices="
                  << plan.local_tp_devices.size() << ", global_tp=" << plan.usesGlobalTP() << ")");
        return nullptr;
    }

    std::unique_ptr<ITPContext> TPContextFactory::createFromDomain(
        const TPDomainParticipation &domain,
        ExecutionDomainScope resolved_scope,
        MPI_Comm base_comm)
    {
        if (domain.devices.empty())
        {
            LOG_ERROR("TPContextFactory::createFromDomain - Domain has no devices");
            return nullptr;
        }

        if (resolved_scope == ExecutionDomainScope::AUTO ||
            (resolved_scope != ExecutionDomainScope::SINGLE && resolved_scope != ExecutionDomainScope::RANK_LOCAL &&
             resolved_scope != ExecutionDomainScope::NODE_LOCAL && resolved_scope != ExecutionDomainScope::GLOBAL))
            throw std::invalid_argument("TP domain creation requires canonical resolved execution scope");
        if (resolved_scope == ExecutionDomainScope::SINGLE && domain.devices.size() != 1)
            throw std::invalid_argument("Single-device TP scope cannot name multiple participants");

        // Cross-rank does not imply cross-node. GlobalTPContext authenticates
        // the physical relationship from its actual admitted communicator.
        if (resolved_scope == ExecutionDomainScope::NODE_LOCAL || resolved_scope == ExecutionDomainScope::GLOBAL)
        {
            LOG_DEBUG("TPContextFactory::createFromDomain - Domain '" << domain.domain_name
                                                                      << "' is GLOBAL, creating GlobalTPContext");

            // For global domains, all ranks with this domain_id participate
            // Use domain_id as the MPI_Comm_split color
            return GlobalTPContext::createWithSplit(
                base_comm,
                domain.domain_id,
                domain.domain_id,         // color = domain_id
                domain.my_index_in_domain, // key = my index
                domain.devices.at(domain.my_index_in_domain),
                "",
                domain.backend
            );
        }

        // Local domain - all devices are within this rank
        LOG_DEBUG("TPContextFactory::createFromDomain - Domain '" << domain.domain_name
                                                                  << "' is LOCAL, creating LocalTPContext");

        return createLocal(domain.devices, domain.weights, domain.backend);
    }

    // =============================================================================
    // Explicit Local TP Creation
    // =============================================================================

    std::unique_ptr<ILocalTPContext> TPContextFactory::createLocal(
        const std::vector<GlobalDeviceAddress> &devices,
        const std::vector<float> &weights,
        CollectiveBackendType backend)
    {
        if (devices.empty())
        {
            LOG_ERROR("TPContextFactory::createLocal - devices vector is empty");
            return nullptr;
        }

        if (devices.size() == 1)
        {
            LOG_WARN("TPContextFactory::createLocal - Single device, LocalTPContext is trivial");
        }

        LOG_DEBUG("TPContextFactory::createLocal - Creating LocalTPContext with "
                  << devices.size() << " devices");

        try
        {
            return std::make_unique<LocalTPContext>(devices, weights, backend);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("TPContextFactory::createLocal - Failed to create LocalTPContext: " << e.what());
            return nullptr;
        }
    }

    std::unique_ptr<ILocalTPContext> TPContextFactory::createLocalFromPlan(
        const RankExecutionPlan &plan)
    {
        if (!plan.usesLocalTP())
        {
            LOG_DEBUG("TPContextFactory::createLocalFromPlan - Plan has no local TP configured");
            return nullptr;
        }

        LOG_DEBUG("TPContextFactory::createLocalFromPlan - Creating LocalTPContext from plan with "
                  << plan.local_tp_devices.size() << " devices");

        return createLocal(
            plan.local_tp_devices,
            plan.local_tp_weights,
            plan.local_tp_backend);
    }

    // =============================================================================
    // Explicit Global TP Creation
    // =============================================================================

    std::unique_ptr<IGlobalTPContext> TPContextFactory::createGlobal(
        MPI_Comm base_comm,
        int domain_id,
        int color,
        int key,
        std::optional<GlobalDeviceAddress> local_device,
        const std::string &hostfile_path,
        CollectiveBackendType backend)
    {
        if (base_comm == MPI_COMM_NULL)
        {
            LOG_ERROR("TPContextFactory::createGlobal - base_comm is MPI_COMM_NULL");
            return nullptr;
        }

        LOG_DEBUG("TPContextFactory::createGlobal - Creating GlobalTPContext with domain_id="
                  << domain_id << ", color=" << color << ", key=" << key);

        return GlobalTPContext::createWithSplit(base_comm, domain_id, color, key,
                                               std::move(local_device), hostfile_path, backend);
    }

    std::unique_ptr<IGlobalTPContext> TPContextFactory::createGlobalFromPlan(
        const RankExecutionPlan &plan,
        MPI_Comm base_comm,
        const std::string &hostfile_path)
    {
        if (!plan.usesGlobalTP())
        {
            LOG_DEBUG("TPContextFactory::createGlobalFromPlan - Plan has no global TP configured");
            return nullptr;
        }

        if (!plan.global_tp_domain_id.has_value())
        {
            LOG_ERROR("TPContextFactory::createGlobalFromPlan - Plan claims global TP but domain_id not set");
            return nullptr;
        }

        int domain_id = plan.global_tp_domain_id.value();

        LOG_DEBUG("TPContextFactory::createGlobalFromPlan - Creating GlobalTPContext: "
                  << "domain_id=" << domain_id
                  << ", rank_in_domain=" << plan.global_tp_rank_in_domain
                  << ", domain_size=" << plan.global_tp_domain_size);

        // Use domain_id as MPI_Comm_split color so all ranks with same domain join
        // Use rank_in_domain as key to preserve ordering within domain
        return GlobalTPContext::createWithSplit(
            base_comm,
            domain_id,
            domain_id,                     // color
            plan.global_tp_rank_in_domain, // key
            plan.primary_device,
            hostfile_path);
    }

} // namespace llaminar2
