#include "MoEExpertOverlayPreparationPlan.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
namespace
{
    const char *residencyPolicyName(ExpertResidencyPolicy policy)
    {
        switch (policy)
        {
        case ExpertResidencyPolicy::Disabled: return "Disabled";
        case ExpertResidencyPolicy::StaticById: return "StaticById";
        case ExpertResidencyPolicy::HistogramTieredCache: return "HistogramTieredCache";
        case ExpertResidencyPolicy::ExplicitMasks: return "ExplicitMasks";
        }
        return "Unknown";
    }

    MoEExpertOverlayDomainPreparationStats &statsFor(
        MoEExpertOverlayPreparationDiagnostics &diagnostics,
        const std::string &domain_name,
        DeviceId device,
        ExpertResidencyPolicy residency_policy)
    {
        auto it = std::find_if(diagnostics.domains.begin(), diagnostics.domains.end(),
                               [&](const auto &stats) {
                                   return stats.domain_name == domain_name && stats.device == device;
                               });
        if (it != diagnostics.domains.end())
            return *it;

        MoEExpertOverlayDomainPreparationStats stats;
        stats.domain_name = domain_name;
        stats.device = device;
        stats.residency_policy = residency_policy;
        stats.accelerator = device.is_gpu();
        diagnostics.domains.push_back(std::move(stats));
        return diagnostics.domains.back();
    }

    std::vector<DeviceId> preparationDevicesFor(const MoEOverlayRuntimeDomain &domain)
    {
        std::vector<DeviceId> devices;
        for (const auto &participant : domain.participants)
        {
            if (participant.local_device.is_valid())
                devices.push_back(participant.local_device);
        }
        if (devices.empty() && domain.primary_device.is_valid())
            devices.push_back(domain.primary_device);

        std::sort(devices.begin(), devices.end());
        devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
        return devices;
    }
} // namespace

    const MoEExpertOverlayDomainPreparationStats *MoEExpertOverlayPreparationDiagnostics::domainStats(
        const std::string &domain_name,
        DeviceId device) const
    {
        auto it = std::find_if(domains.begin(), domains.end(),
                               [&](const auto &stats) {
                                   return stats.domain_name == domain_name && stats.device == device;
                               });
        return it == domains.end() ? nullptr : &(*it);
    }

    std::string MoEExpertOverlayPreparationDiagnostics::render() const
    {
        std::ostringstream out;
        out << "MoE expert overlay preparation diagnostics:";
        if (domains.empty())
        {
            out << " no domains";
            return out.str();
        }

        for (const auto &stats : domains)
        {
            out << "\n  domain " << stats.domain_name
                << " device=" << stats.device.to_string()
                << " residency=" << residencyPolicyName(stats.residency_policy)
                << " accelerator=" << (stats.accelerator ? "true" : "false")
                << " fallback=" << (stats.fallback ? "true" : "false")
                << " assigned_experts=" << stats.assigned_routed_experts
                << " planned_engines=" << stats.planned_engine_count
                << " estimated_routed_bytes=" << stats.estimated_routed_bytes
                << " memory_budget_bytes=" << stats.memory_budget_bytes;
        }
        return out.str();
    }

    MoEExpertOverlayPreparationPlan MoEExpertOverlayPreparationPlan::build(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        size_t routed_expert_bytes_per_expert)
    {
        const auto &source = runtime_plan.sourcePlan();
        if (!source.isTieredOverlay())
            return {};

        MoEExpertOverlayPreparationPlan result;
        constexpr ExpertGemmRegistry::WeightRole kRoles[] = {
            ExpertGemmRegistry::WeightRole::GATE,
            ExpertGemmRegistry::WeightRole::UP,
            ExpertGemmRegistry::WeightRole::DOWN,
        };

        for (size_t tier_index = 0; tier_index < source.routed_tiers.size(); ++tier_index)
        {
            const auto &tier = source.routed_tiers[tier_index];
            const auto &domain = runtime_plan.domainForTier(tier_index);
            for (const auto device : preparationDevicesFor(domain))
            {
                auto &stats = statsFor(result.diagnostics_, tier.domain, device, source.residency_policy);
                stats.fallback = stats.fallback || tier.fallback;
                stats.memory_budget_bytes = std::max(stats.memory_budget_bytes, tier.memory_budget_bytes);
            }
        }

        for (const auto &placement : source.placements)
        {
            for (size_t expert_index = 0; expert_index < placement.routed_expert_tier.size(); ++expert_index)
            {
                const int tier_index = placement.routed_expert_tier[expert_index];
                if (tier_index < 0 || tier_index >= static_cast<int>(source.routed_tiers.size()))
                {
                    std::ostringstream message;
                    message << "MoE expert overlay preparation plan has invalid tier "
                            << tier_index << " for layer " << placement.layer
                            << " expert " << expert_index;
                    throw std::runtime_error(message.str());
                }

                const auto &tier = source.routed_tiers[static_cast<size_t>(tier_index)];
                const auto &domain = runtime_plan.domainForTier(static_cast<size_t>(tier_index));
                for (const auto device : preparationDevicesFor(domain))
                {
                    auto &stats = statsFor(result.diagnostics_, tier.domain, device, source.residency_policy);
                    ++stats.assigned_routed_experts;
                    stats.planned_engine_count += 3;
                    stats.estimated_routed_bytes += routed_expert_bytes_per_expert;
                    stats.fallback = stats.fallback || tier.fallback;
                    stats.memory_budget_bytes = std::max(stats.memory_budget_bytes, tier.memory_budget_bytes);

                    for (const auto role : kRoles)
                    {
                        MoEExpertOverlayPreparationRequest request;
                        request.layer = placement.layer;
                        request.expert_id = static_cast<int>(expert_index);
                        request.role = role;
                        request.tier_index = tier_index;
                        request.tier_name = tier.name;
                        request.domain_name = tier.domain;
                        request.device = device;
                        request.residency_policy = source.residency_policy;
                        request.fallback = tier.fallback;
                        result.requests_.push_back(std::move(request));
                    }
                }
            }
        }

        std::sort(result.diagnostics_.domains.begin(), result.diagnostics_.domains.end(),
                  [](const auto &lhs, const auto &rhs) {
                      if (lhs.domain_name != rhs.domain_name)
                          return lhs.domain_name < rhs.domain_name;
                      return lhs.device < rhs.device;
                  });

        return result;
    }

    bool MoEExpertOverlayPreparationPlan::hasRequestsForDevice(DeviceId device) const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [&](const auto &request) { return request.device == device; });
    }

    bool MoEExpertOverlayPreparationPlan::hasAcceleratorRequests() const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [](const auto &request) { return request.device.is_gpu(); });
    }

    bool MoEExpertOverlayPreparationPlan::hasCpuRoutedAssignments() const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [](const auto &request) { return request.device.is_cpu(); });
    }

    std::vector<DeviceId> MoEExpertOverlayPreparationPlan::acceleratorDevices() const
    {
        std::set<DeviceId> devices;
        for (const auto &request : requests_)
        {
            if (request.device.is_gpu())
                devices.insert(request.device);
        }
        return {devices.begin(), devices.end()};
    }

    bool MoEExpertOverlayPreparationPlan::shouldPrepare(
        DeviceId device,
        int layer,
        int expert_id,
        ExpertGemmRegistry::WeightRole role) const
    {
        return requestFor(device, layer, expert_id, role) != nullptr;
    }

    const MoEExpertOverlayPreparationRequest *MoEExpertOverlayPreparationPlan::requestFor(
        DeviceId device,
        int layer,
        int expert_id,
        ExpertGemmRegistry::WeightRole role) const
    {
        auto it = std::find_if(requests_.begin(), requests_.end(),
                               [&](const auto &request) {
                                   return request.device == device &&
                                          request.layer == layer &&
                                          request.expert_id == expert_id &&
                                          request.role == role &&
                                          request.device.is_gpu();
                               });
        return it == requests_.end() ? nullptr : &(*it);
    }

    bool MoEExpertOverlayPreparationPlan::hasAnyRequestForDeviceLayerRole(
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [&](const auto &request) {
                               return request.device == device &&
                                      request.layer == layer &&
                                      request.role == role &&
                                      request.device.is_gpu();
                           });
    }

    std::vector<std::string> MoEExpertOverlayPreparationPlan::domainsForDeviceLayerRole(
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        std::vector<std::string> domains;
        for (const auto &request : requests_)
        {
            if (request.device == device &&
                request.layer == layer &&
                request.role == role &&
                request.device.is_gpu())
            {
                domains.push_back(request.domain_name);
            }
        }
        std::sort(domains.begin(), domains.end());
        domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
        return domains;
    }

    std::vector<int> MoEExpertOverlayPreparationPlan::expertsForDeviceLayerRole(
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        std::vector<int> experts;
        for (const auto &request : requests_)
        {
            if (request.device == device &&
                request.layer == layer &&
                request.role == role &&
                request.device.is_gpu())
            {
                experts.push_back(request.expert_id);
            }
        }
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        return experts;
    }

    std::vector<int> MoEExpertOverlayPreparationPlan::expertsForDomainDeviceLayerRole(
        const std::string &domain_name,
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        std::vector<int> experts;
        for (const auto &request : requests_)
        {
            if (request.domain_name == domain_name &&
                request.device == device &&
                request.layer == layer &&
                request.role == role &&
                request.device.is_gpu())
            {
                experts.push_back(request.expert_id);
            }
        }
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        return experts;
    }

} // namespace llaminar2
