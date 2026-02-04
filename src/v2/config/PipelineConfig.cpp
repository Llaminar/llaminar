/**
 * @file PipelineConfig.cpp
 * @brief Implementation of PipelineConfig
 *
 * Part of the Unified PP Graph Architecture (Phase 1.3).
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "config/PipelineConfig.h"
#include "config/BackendSelector.h"
#include <algorithm>
#include <set>
#include <sstream>

namespace llaminar2
{

// =============================================================================
// Lookup Methods
// =============================================================================

const TPDomainConfig *PipelineConfig::getDomain(const std::string &name) const
{
    for (const auto &domain : tp_domains)
    {
        if (domain.name == name)
        {
            return &domain;
        }
    }
    return nullptr;
}

const TPDomainConfig *PipelineConfig::getDomainForStage(int stage_id) const
{
    for (const auto &stage : pp_stages)
    {
        if (stage.stage_id == stage_id)
        {
            return getDomain(stage.domain_name);
        }
    }
    return nullptr;
}

const PPStageConfig *PipelineConfig::getStageForLayer(int layer_idx) const
{
    for (const auto &stage : pp_stages)
    {
        if (stage.containsLayer(layer_idx))
        {
            return &stage;
        }
    }
    return nullptr;
}

DeviceId PipelineConfig::getDeviceForLayer(int layer_idx) const
{
    const auto *stage = getStageForLayer(layer_idx);
    if (!stage)
    {
        return DeviceId::cpu();
    }

    const auto *domain = getDomain(stage->domain_name);
    if (!domain)
    {
        return DeviceId::cpu();
    }

    return domain->primaryDevice();
}

bool PipelineConfig::needsPPTransfer(int from_layer, int to_layer) const
{
    int from_stage = getStageIdForLayer(from_layer);
    int to_stage = getStageIdForLayer(to_layer);

    // If either layer is not found, no transfer needed (or invalid request)
    if (from_stage < 0 || to_stage < 0)
    {
        return false;
    }

    return from_stage != to_stage;
}

CollectiveBackendType PipelineConfig::getTransferBackend(int from_stage, int to_stage) const
{
    auto it = pp_transfer_backends.find({from_stage, to_stage});
    if (it != pp_transfer_backends.end())
    {
        return it->second;
    }
    return CollectiveBackendType::AUTO;
}

int PipelineConfig::getStageIdForLayer(int layer_idx) const
{
    const auto *stage = getStageForLayer(layer_idx);
    if (stage)
    {
        return stage->stage_id;
    }
    return -1;
}

// =============================================================================
// Query Methods
// =============================================================================

int PipelineConfig::numStages() const
{
    return static_cast<int>(pp_stages.size());
}

int PipelineConfig::maxTPDegree() const
{
    int max_degree = 0;
    for (const auto &domain : tp_domains)
    {
        max_degree = std::max(max_degree, domain.degree());
    }
    return max_degree;
}

bool PipelineConfig::isSingleStage() const
{
    return pp_stages.size() == 1;
}

bool PipelineConfig::hasTP() const
{
    for (const auto &domain : tp_domains)
    {
        if (domain.degree() > 1)
        {
            return true;
        }
    }
    return false;
}

bool PipelineConfig::hasPP() const
{
    return pp_stages.size() > 1;
}

std::vector<DeviceId> PipelineConfig::getAllDevices() const
{
    std::vector<DeviceId> result;

    for (const auto &domain : tp_domains)
    {
        for (const auto &device : domain.devices)
        {
            // Check for duplicates (preserve order, first occurrence wins)
            bool found = false;
            for (const auto &existing : result)
            {
                if (existing == device)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                result.push_back(device);
            }
        }
    }

    return result;
}

// =============================================================================
// Validation
// =============================================================================

bool PipelineConfig::validate(std::string *error_msg) const
{
    auto setError = [error_msg](const std::string &msg)
    {
        if (error_msg)
        {
            *error_msg = msg;
        }
    };

    // Check total_layers > 0
    if (total_layers <= 0)
    {
        setError("total_layers must be > 0, got " + std::to_string(total_layers));
        return false;
    }

    // Check at least one domain
    if (tp_domains.empty())
    {
        setError("No TP domains defined");
        return false;
    }

    // Check at least one stage
    if (pp_stages.empty())
    {
        setError("No PP stages defined");
        return false;
    }

    // Validate each domain
    for (size_t i = 0; i < tp_domains.size(); ++i)
    {
        std::string domain_error;
        if (!tp_domains[i].validate(&domain_error))
        {
            setError("Domain " + std::to_string(i) + " (" + tp_domains[i].name + ") invalid: " + domain_error);
            return false;
        }
    }

    // Validate each stage and check domain references
    int embedding_count = 0;
    int lm_head_count = 0;

    for (size_t i = 0; i < pp_stages.size(); ++i)
    {
        const auto &stage = pp_stages[i];

        // Check domain reference
        if (!getDomain(stage.domain_name))
        {
            setError("Stage " + std::to_string(i) + " references unknown domain: " + stage.domain_name);
            return false;
        }

        // Validate stage
        std::string stage_error;
        if (!stage.validate(total_layers, &stage_error))
        {
            setError("Stage " + std::to_string(i) + " invalid: " + stage_error);
            return false;
        }

        // Count embedding/lm_head stages
        if (stage.has_embedding)
        {
            embedding_count++;
        }
        if (stage.has_lm_head)
        {
            lm_head_count++;
        }
    }

    // Check exactly one embedding stage
    if (embedding_count != 1)
    {
        setError("Expected exactly 1 stage with has_embedding=true, got " + std::to_string(embedding_count));
        return false;
    }

    // Check exactly one LM head stage
    if (lm_head_count != 1)
    {
        setError("Expected exactly 1 stage with has_lm_head=true, got " + std::to_string(lm_head_count));
        return false;
    }

    // Check layer coverage: no gaps, no overlaps
    // Build a sorted list of layer ranges
    std::vector<std::pair<int, int>> ranges;
    ranges.reserve(pp_stages.size());
    for (const auto &stage : pp_stages)
    {
        ranges.push_back({stage.first_layer, stage.last_layer});
    }
    std::sort(ranges.begin(), ranges.end());

    // Check coverage starts at 0
    if (ranges.empty() || ranges[0].first != 0)
    {
        setError("Layer coverage does not start at 0");
        return false;
    }

    // Check for gaps and overlaps
    for (size_t i = 1; i < ranges.size(); ++i)
    {
        if (ranges[i].first < ranges[i - 1].second)
        {
            setError("Layer overlap detected: layer " + std::to_string(ranges[i].first) +
                     " covered by multiple stages");
            return false;
        }
        if (ranges[i].first > ranges[i - 1].second)
        {
            setError("Layer gap detected: layers " + std::to_string(ranges[i - 1].second) +
                     " to " + std::to_string(ranges[i].first - 1) + " not covered");
            return false;
        }
    }

    // Check coverage ends at total_layers
    if (ranges.back().second != total_layers)
    {
        setError("Layer coverage does not end at total_layers (" +
                 std::to_string(total_layers) + "), ends at " +
                 std::to_string(ranges.back().second));
        return false;
    }

    return true;
}

// =============================================================================
// Auto-Completion Methods
// =============================================================================

void PipelineConfig::autoSelectBackends()
{
    // 1. Auto-select TP domain backends
    for (auto &domain : tp_domains)
    {
        if (domain.tp_backend == CollectiveBackendType::AUTO)
        {
            domain.tp_backend = BackendSelector::selectForTPDomain(domain.devices);
        }
    }

    // 2. Auto-select PP transfer backends for adjacent stages
    if (pp_stages.size() < 2)
    {
        return; // No transfers needed for single-stage
    }

    for (size_t i = 0; i + 1 < pp_stages.size(); ++i)
    {
        int from_stage = pp_stages[i].stage_id;
        int to_stage = pp_stages[i + 1].stage_id;

        auto key = std::make_pair(from_stage, to_stage);

        // Skip if already explicitly set (and not AUTO)
        auto it = pp_transfer_backends.find(key);
        if (it != pp_transfer_backends.end() &&
            it->second != CollectiveBackendType::AUTO)
        {
            continue;
        }

        // Get primary devices for each stage's domain
        const TPDomainConfig *from_domain = getDomainForStage(from_stage);
        const TPDomainConfig *to_domain = getDomainForStage(to_stage);

        if (!from_domain || !to_domain)
        {
            // Invalid config, will fail validation later
            continue;
        }

        DeviceId from_device = from_domain->primaryDevice();
        DeviceId to_device = to_domain->primaryDevice();

        // Select backend based on device types
        pp_transfer_backends[key] = BackendSelector::selectForTransfer(from_device, to_device);
    }
}

bool PipelineConfig::completeAndValidate(std::string *error_msg)
{
    autoSelectBackends();
    return validate(error_msg);
}

// =============================================================================
// Factory Methods
// =============================================================================

PipelineConfig PipelineConfig::singleDevice(int num_layers, DeviceId device)
{
    PipelineConfig config;
    config.total_layers = num_layers;

    // Create a single domain with one device
    TPDomainConfig domain;
    domain.name = "default";
    domain.devices.push_back(device);
    domain.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain));

    // Create a single stage covering all layers
    config.pp_stages.push_back(PPStageConfig::fullModel(num_layers, "default"));

    return config;
}

PipelineConfig PipelineConfig::tensorParallel(int num_layers,
                                               const std::vector<DeviceId> &devices,
                                               CollectiveBackendType backend)
{
    PipelineConfig config;
    config.total_layers = num_layers;

    // Create a single domain with all devices
    TPDomainConfig domain;
    domain.name = "tp_domain";
    domain.devices = devices;
    domain.tp_backend = backend;
    config.tp_domains.push_back(std::move(domain));

    // Create a single stage covering all layers
    config.pp_stages.push_back(PPStageConfig::fullModel(num_layers, "tp_domain"));

    return config;
}

PipelineConfig PipelineConfig::pipelineParallel2Stage(int num_layers,
                                                       DeviceId device0, int split_layer,
                                                       DeviceId device1,
                                                       CollectiveBackendType transfer_backend)
{
    PipelineConfig config;
    config.total_layers = num_layers;

    // Create two domains, one per stage
    TPDomainConfig domain0;
    domain0.name = "stage0_domain";
    domain0.devices.push_back(device0);
    domain0.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain0));

    TPDomainConfig domain1;
    domain1.name = "stage1_domain";
    domain1.devices.push_back(device1);
    domain1.tp_backend = CollectiveBackendType::AUTO;
    config.tp_domains.push_back(std::move(domain1));

    // Create two stages
    config.pp_stages.push_back(PPStageConfig::firstStage(0, "stage0_domain", 0, split_layer));
    config.pp_stages.push_back(PPStageConfig::lastStage(1, "stage1_domain", split_layer, num_layers));

    // Set the transfer backend
    config.pp_transfer_backends[{0, 1}] = transfer_backend;

    return config;
}

} // namespace llaminar2
