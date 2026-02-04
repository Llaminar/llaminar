/**
 * @file TPContextFactory.cpp
 * @brief Implementation of factory for creating ITPContext instances
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

namespace llaminar2 {

// =============================================================================
// Main Factory Methods
// =============================================================================

std::unique_ptr<ITPContext> TPContextFactory::create(
    const RankExecutionPlan& plan,
    MPI_Comm base_comm)
{
    // Global TP takes precedence (encompasses local as a special case)
    if (plan.usesGlobalTP()) {
        LOG_DEBUG("TPContextFactory::create - Plan uses Global TP, creating GlobalTPContext");
        return createGlobalFromPlan(plan, base_comm);
    }
    
    // Local TP only
    if (plan.usesLocalTP()) {
        LOG_DEBUG("TPContextFactory::create - Plan uses Local TP, creating LocalTPContext");
        return createLocalFromPlan(plan);
    }
    
    // No TP configured
    LOG_DEBUG("TPContextFactory::create - No TP configured in plan (local_tp_devices=" 
              << plan.local_tp_devices.size() << ", global_tp=" << plan.usesGlobalTP() << ")");
    return nullptr;
}

std::unique_ptr<ITPContext> TPContextFactory::createFromDomain(
    const TPDomainParticipation& domain,
    MPI_Comm base_comm)
{
    if (domain.devices.empty()) {
        LOG_ERROR("TPContextFactory::createFromDomain - Domain has no devices");
        return nullptr;
    }
    
    // Check if this is a global domain by:
    // 1. Backend type (UPI/MPI implies cross-rank)
    // 2. Multiple devices with different hostnames
    bool is_global = false;
    
    // Check if backend suggests cross-rank communication
    if (domain.backend == CollectiveBackendType::UPI || 
        domain.backend == CollectiveBackendType::MPI) {
        is_global = true;
    }
    
    // Check if devices span multiple hostnames (indicates cross-rank)
    if (!is_global && domain.devices.size() > 1) {
        const std::string& first_hostname = domain.devices[0].hostname;
        for (size_t i = 1; i < domain.devices.size(); ++i) {
            if (domain.devices[i].hostname != first_hostname) {
                is_global = true;
                break;
            }
        }
    }
    
    if (is_global) {
        LOG_DEBUG("TPContextFactory::createFromDomain - Domain '" << domain.domain_name 
                  << "' is GLOBAL, creating GlobalTPContext");
        
        // For global domains, all ranks with this domain_id participate
        // Use domain_id as the MPI_Comm_split color
        return GlobalTPContext::createWithSplit(
            base_comm,
            domain.domain_id,
            domain.domain_id,  // color = domain_id
            domain.my_index_in_domain  // key = my index
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
    const std::vector<GlobalDeviceAddress>& devices,
    const std::vector<float>& weights,
    CollectiveBackendType backend)
{
    if (devices.empty()) {
        LOG_ERROR("TPContextFactory::createLocal - devices vector is empty");
        return nullptr;
    }
    
    if (devices.size() == 1) {
        LOG_WARN("TPContextFactory::createLocal - Single device, LocalTPContext is trivial");
    }
    
    LOG_DEBUG("TPContextFactory::createLocal - Creating LocalTPContext with " 
              << devices.size() << " devices");
    
    try {
        return std::make_unique<LocalTPContext>(devices, weights, backend);
    } catch (const std::exception& e) {
        LOG_ERROR("TPContextFactory::createLocal - Failed to create LocalTPContext: " << e.what());
        return nullptr;
    }
}

std::unique_ptr<ILocalTPContext> TPContextFactory::createLocalFromPlan(
    const RankExecutionPlan& plan)
{
    if (!plan.usesLocalTP()) {
        LOG_DEBUG("TPContextFactory::createLocalFromPlan - Plan has no local TP configured");
        return nullptr;
    }
    
    LOG_DEBUG("TPContextFactory::createLocalFromPlan - Creating LocalTPContext from plan with "
              << plan.local_tp_devices.size() << " devices");
    
    return createLocal(
        plan.local_tp_devices,
        plan.local_tp_weights,
        plan.local_tp_backend
    );
}

// =============================================================================
// Explicit Global TP Creation
// =============================================================================

std::unique_ptr<IGlobalTPContext> TPContextFactory::createGlobal(
    MPI_Comm base_comm,
    int domain_id,
    int color,
    int key)
{
    if (base_comm == MPI_COMM_NULL) {
        LOG_ERROR("TPContextFactory::createGlobal - base_comm is MPI_COMM_NULL");
        return nullptr;
    }
    
    LOG_DEBUG("TPContextFactory::createGlobal - Creating GlobalTPContext with domain_id=" 
              << domain_id << ", color=" << color << ", key=" << key);
    
    return GlobalTPContext::createWithSplit(base_comm, domain_id, color, key);
}

std::unique_ptr<IGlobalTPContext> TPContextFactory::createGlobalFromPlan(
    const RankExecutionPlan& plan,
    MPI_Comm base_comm)
{
    if (!plan.usesGlobalTP()) {
        LOG_DEBUG("TPContextFactory::createGlobalFromPlan - Plan has no global TP configured");
        return nullptr;
    }
    
    if (!plan.global_tp_domain_id.has_value()) {
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
        domain_id,  // color
        plan.global_tp_rank_in_domain  // key
    );
}

} // namespace llaminar2
