/**
 * @file OrchestrationConfig.cpp
 * @brief Implementation of OrchestrationConfig and related structures
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationConfig.h"
#include "config/ConfigValidator.h"
#include "execution/parallelism_tree/ParallelismTree.h"
#include "utils/Logger.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <cmath>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace llaminar2
{

    // =========================================================================
    // Enum to String Conversions
    // =========================================================================

    const char *tpScopeToString(TPScope scope)
    {
        switch (scope)
        {
        case TPScope::AUTO:
            return "auto";
        case TPScope::LOCAL:
            return "local";
        case TPScope::NODE_LOCAL:
            return "node_local";
        case TPScope::GLOBAL:
            return "global";
        case TPScope::HYBRID:
            return "hybrid";
        default:
            return "unknown";
        }
    }

    const char *deviceAssignmentModeToString(DeviceAssignmentMode mode)
    {
        switch (mode)
        {
        case DeviceAssignmentMode::AUTO:
            return "auto";
        case DeviceAssignmentMode::LOCAL_GPU:
            return "local_gpu";
        case DeviceAssignmentMode::ROUND_ROBIN:
            return "round_robin";
        case DeviceAssignmentMode::EXPLICIT:
            return "explicit";
        default:
            return "unknown";
        }
    }

    const char *ppSplitModeToString(PPSplitMode mode)
    {
        switch (mode)
        {
        case PPSplitMode::EQUAL:
            return "equal";
        case PPSplitMode::WEIGHTED:
            return "weighted";
        case PPSplitMode::MANUAL:
            return "manual";
        default:
            return "unknown";
        }
    }

    const char *mpiProfileToString(MPIProfile profile)
    {
        switch (profile)
        {
        case MPIProfile::AUTO:
            return "auto";
        case MPIProfile::TUNED:
            return "tuned";
        default:
            return "unknown";
        }
    }

    // collectiveBackendTypeToString - now in CollectiveBackendType.cpp

    // =========================================================================
    // String to Enum Parsing
    // =========================================================================

    namespace
    {
        std::string toLower(const std::string &str)
        {
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return lower;
        }

        std::vector<std::string> split(const std::string &str, char delim)
        {
            std::vector<std::string> parts;
            std::stringstream ss(str);
            std::string part;
            while (std::getline(ss, part, delim))
            {
                // Trim whitespace
                size_t start = part.find_first_not_of(" \t");
                size_t end = part.find_last_not_of(" \t");
                if (start != std::string::npos)
                {
                    parts.push_back(part.substr(start, end - start + 1));
                }
            }
            return parts;
        }
    } // anonymous namespace

    std::optional<TPScope> parseTpScope(const std::string &str)
    {
        std::string lower = toLower(str);
        if (lower == "auto")
            return TPScope::AUTO;
        if (lower == "local")
            return TPScope::LOCAL;
        if (lower == "node_local" || lower == "nodelocal")
            return TPScope::NODE_LOCAL;
        if (lower == "global")
            return TPScope::GLOBAL;
        if (lower == "hybrid")
            return TPScope::HYBRID;
        return std::nullopt;
    }

    std::optional<DeviceAssignmentMode> parseDeviceAssignmentMode(const std::string &str)
    {
        std::string lower = toLower(str);
        if (lower == "auto")
            return DeviceAssignmentMode::AUTO;
        if (lower == "local_gpu" || lower == "local-gpu")
            return DeviceAssignmentMode::LOCAL_GPU;
        if (lower == "round_robin" || lower == "round-robin")
            return DeviceAssignmentMode::ROUND_ROBIN;
        if (lower == "explicit")
            return DeviceAssignmentMode::EXPLICIT;
        return std::nullopt;
    }

    std::optional<PPSplitMode> parsePpSplitMode(const std::string &str)
    {
        std::string lower = toLower(str);
        if (lower == "equal")
            return PPSplitMode::EQUAL;
        if (lower == "weighted")
            return PPSplitMode::WEIGHTED;
        if (lower == "manual")
            return PPSplitMode::MANUAL;
        return std::nullopt;
    }

    std::optional<MPIProfile> parseMPIProfile(const std::string &str)
    {
        std::string lower = toLower(str);
        if (lower == "auto")
            return MPIProfile::AUTO;
        if (lower == "tuned")
            return MPIProfile::TUNED;
        return std::nullopt;
    }

    // parseCollectiveBackendType - now in CollectiveBackendType.cpp

    // =========================================================================
    // DomainDefinition Implementation
    // =========================================================================

    namespace
    {
        ExecutionDomainScope toExecutionDomainScope(TPScope scope)
        {
            switch (scope)
            {
            case TPScope::LOCAL:
                return ExecutionDomainScope::LOCAL;
            case TPScope::NODE_LOCAL:
                return ExecutionDomainScope::NODE_LOCAL;
            case TPScope::GLOBAL:
                return ExecutionDomainScope::GLOBAL;
            case TPScope::AUTO:
            case TPScope::HYBRID:
                return ExecutionDomainScope::AUTO;
            }
            return ExecutionDomainScope::AUTO;
        }

        TPScope toTPScope(ExecutionDomainScope scope)
        {
            switch (scope)
            {
            case ExecutionDomainScope::LOCAL:
                return TPScope::LOCAL;
            case ExecutionDomainScope::NODE_LOCAL:
                return TPScope::NODE_LOCAL;
            case ExecutionDomainScope::GLOBAL:
                return TPScope::GLOBAL;
            case ExecutionDomainScope::SINGLE:
            case ExecutionDomainScope::AUTO:
                return TPScope::AUTO;
            }
            return TPScope::AUTO;
        }

        bool sameWeights(const std::vector<float> &lhs, const std::vector<float> &rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t index = 0; index < lhs.size(); ++index)
            {
                if (std::abs(lhs[index] - rhs[index]) > 1e-6f)
                    return false;
            }
            return true;
        }

        bool sameDomainDefinition(
            const ExecutionDomainDefinition &lhs,
            const ExecutionDomainDefinition &rhs)
        {
            auto normalizedScope = [](const ExecutionDomainDefinition &domain)
            {
                if (domain.scope == ExecutionDomainScope::SINGLE && domain.participants.size() == 1)
                    return ExecutionDomainScope::AUTO;
                return domain.scope;
            };

            return lhs.name == rhs.name &&
                   lhs.participants == rhs.participants &&
                   sameWeights(lhs.weights, rhs.weights) &&
                   lhs.backend == rhs.backend &&
                   normalizedScope(lhs) == normalizedScope(rhs) &&
                   lhs.owner_rank == rhs.owner_rank &&
                   lhs.ranks == rhs.ranks &&
                   lhs.compute_kind == rhs.compute_kind;
        }

        void addUniqueName(std::vector<std::string> &names, const std::string &name)
        {
            if (name.empty())
                return;
            if (std::find(names.begin(), names.end(), name) == names.end())
                names.push_back(name);
        }

        std::vector<std::string> overlayDenseDomainNames(const MoEExpertParallelPlan &plan)
        {
            std::vector<std::string> names;
            addUniqueName(names, plan.continuation_domain);
            addUniqueName(names, plan.effectiveBaseModelDomain());
            addUniqueName(names, plan.shared_expert_domain);
            for (const auto &domain : plan.dense_domains)
                addUniqueName(names, domain.name);
            return names;
        }

        std::vector<std::string> overlayRoutedDomainNames(const MoEExpertParallelPlan &plan)
        {
            std::vector<std::string> names;
            for (const auto &tier : plan.routed_tiers)
                addUniqueName(names, tier.domain);
            for (const auto &domain : plan.domains)
                addUniqueName(names, domain.name);
            return names;
        }
    } // namespace

    DomainDefinition DomainDefinition::parse(const std::string &spec)
    {
        auto result = tryParse(spec);
        if (!result)
        {
            throw std::invalid_argument(
                "Failed to parse domain definition: '" + spec + "'. "
                                                                "Expected format: name=device1,device2[;weights=w1,w2][;backend=type]");
        }
        return *result;
    }

    std::optional<DomainDefinition> DomainDefinition::tryParse(const std::string &spec)
    {
        ExecutionDomainParseOptions options;
        options.context = "domain definition";

        auto parsed = ExecutionDomainDefinition::tryParse(spec, options);
        if (!parsed)
            return std::nullopt;
        return fromExecutionDomainDefinition(*parsed);
    }

    ExecutionDomainDefinition DomainDefinition::toExecutionDomainDefinition() const
    {
        ExecutionDomainDefinition domain;
        domain.name = name;
        domain.participants = devices;
        domain.weights = weights;
        domain.backend = backend;
        domain.scope = toExecutionDomainScope(scope);
        domain.owner_rank = owner_rank;
        domain.ranks = explicit_ranks;
        domain.compute_kind = compute_kind;
        return domain;
    }

    DomainDefinition DomainDefinition::fromExecutionDomainDefinition(const ExecutionDomainDefinition &domain)
    {
        DomainDefinition def;
        def.name = domain.name;
        def.devices = domain.participants;
        def.weights = domain.weights;
        def.backend = domain.backend;
        def.compute_kind = domain.compute_kind;
        def.scope = toTPScope(domain.scope);
        def.owner_rank = domain.owner_rank;
        def.explicit_ranks = domain.ranks;
        return def;
    }

    std::vector<std::string> DomainDefinition::validate() const
    {
        return toExecutionDomainDefinition().validate();
    }

    std::string DomainDefinition::toString() const
    {
        return toExecutionDomainDefinition().toString();
    }

    // =========================================================================
    // PPStageDefinition Implementation
    // =========================================================================

    PPStageDefinition PPStageDefinition::parse(const std::string &spec)
    {
        auto result = tryParse(spec);
        if (!result)
        {
            throw std::invalid_argument(
                "Failed to parse PP stage definition: '" + spec + "'. "
                                                                  "Expected format: stage_id=domain_name:first_layer-last_layer");
        }
        return *result;
    }

    std::optional<PPStageDefinition> PPStageDefinition::tryParse(const std::string &spec)
    {
        if (spec.empty())
        {
            return std::nullopt;
        }

        PPStageDefinition def;

        // Format: "stage_id=domain_name:first_layer-last_layer"
        size_t eq_pos = spec.find('=');
        if (eq_pos == std::string::npos)
        {
            return std::nullopt;
        }

        // Parse stage_id
        try
        {
            def.stage_id = std::stoi(spec.substr(0, eq_pos));
        }
        catch (const std::exception &)
        {
            return std::nullopt;
        }

        // Parse "domain_name:first_layer-last_layer"
        std::string rest = spec.substr(eq_pos + 1);
        size_t colon_pos = rest.find(':');
        if (colon_pos == std::string::npos)
        {
            return std::nullopt;
        }

        def.domain_name = rest.substr(0, colon_pos);
        if (def.domain_name.empty())
        {
            return std::nullopt;
        }

        // Parse "first_layer-last_layer"
        std::string layers = rest.substr(colon_pos + 1);
        size_t dash_pos = layers.find('-');
        if (dash_pos == std::string::npos)
        {
            return std::nullopt;
        }

        try
        {
            def.first_layer = std::stoi(layers.substr(0, dash_pos));
            def.last_layer = std::stoi(layers.substr(dash_pos + 1));
        }
        catch (const std::exception &)
        {
            return std::nullopt;
        }

        return def;
    }

    std::vector<std::string> PPStageDefinition::validate() const
    {
        std::vector<std::string> errors;

        if (stage_id < 0)
        {
            errors.push_back("PP stage " + std::to_string(stage_id) + " has invalid ID (must be >= 0)");
        }

        if (domain_name.empty())
        {
            errors.push_back("PP stage " + std::to_string(stage_id) + " has empty domain name");
        }

        if (first_layer < 0)
        {
            errors.push_back("PP stage " + std::to_string(stage_id) + " has invalid first_layer " +
                             std::to_string(first_layer));
        }

        if (last_layer < first_layer)
        {
            errors.push_back("PP stage " + std::to_string(stage_id) + " has last_layer (" +
                             std::to_string(last_layer) + ") < first_layer (" +
                             std::to_string(first_layer) + ")");
        }

        return errors;
    }

    std::string PPStageDefinition::toString() const
    {
        std::ostringstream oss;
        oss << "stage[" << stage_id << "]=" << domain_name
            << " layers[" << first_layer << "-" << last_layer << "]";
        return oss.str();
    }

    // =========================================================================
    // OrchestrationConfig Implementation
    // =========================================================================

    bool OrchestrationConfig::usesNamedDomains() const
    {
        return !domain_definitions.empty() || !pp_stage_definitions.empty();
    }

    std::vector<ExecutionDomainDefinition> OrchestrationConfig::executionDomainDefinitions() const
    {
        std::vector<ExecutionDomainDefinition> domains;
        domains.reserve(domain_definitions.size() +
                        (moe_expert_parallel_plan ? moe_expert_parallel_plan->dense_domains.size() +
                                                        moe_expert_parallel_plan->domains.size()
                                                  : 0));

        std::unordered_map<std::string, size_t> index_by_name;
        auto appendIfNew = [&](const ExecutionDomainDefinition &domain)
        {
            if (index_by_name.find(domain.name) != index_by_name.end())
                return;
            index_by_name.emplace(domain.name, domains.size());
            domains.push_back(domain);
        };

        if (moe_expert_parallel_plan)
        {
            for (const auto &domain : moe_expert_parallel_plan->dense_domains)
                appendIfNew(domain);
            for (const auto &domain : moe_expert_parallel_plan->domains)
                appendIfNew(domain.toExecutionDomainDefinition());
        }

        for (const auto &domain : domain_definitions)
            appendIfNew(domain.toExecutionDomainDefinition());

        return domains;
    }

    std::vector<std::string> normalizeMoEExpertOverlayDomains(OrchestrationConfig &config)
    {
        std::vector<std::string> errors;
        auto &plan_ptr = config.moe_expert_parallel_plan;
        if (!plan_ptr)
            return errors;

        auto &plan = *plan_ptr;
        std::vector<ExecutionDomainDefinition> inventory;
        std::unordered_map<std::string, size_t> index_by_name;

        auto addDomain = [&](const ExecutionDomainDefinition &domain, const std::string &source)
        {
            if (domain.name.empty())
            {
                errors.push_back(source + " defines an execution domain with an empty name");
                return;
            }

            auto existing = index_by_name.find(domain.name);
            if (existing != index_by_name.end())
            {
                auto &prior = inventory[existing->second];
                if (!sameDomainDefinition(prior, domain))
                {
                    errors.push_back("Conflicting execution domain definition for '" + domain.name +
                                     "'. Define each hardware domain once with --define-domain; "
                                     "--moe-expert-overlay-domain is a strict alias and must not "
                                     "redefine a domain with different devices, scope, backend, ranks, weights, or compute kind");
                }
                else if (prior.scope == ExecutionDomainScope::AUTO && domain.scope != ExecutionDomainScope::AUTO)
                {
                    prior = domain;
                }
                return;
            }

            index_by_name.emplace(domain.name, inventory.size());
            inventory.push_back(domain);
        };

        for (const auto &domain : config.domain_definitions)
            addDomain(domain.toExecutionDomainDefinition(), "--define-domain");

        for (const auto &domain : plan.dense_domains)
            addDomain(domain, "MoE continuation dense domain");

        for (const auto &domain : plan.domains)
            addDomain(domain.toExecutionDomainDefinition(), "--moe-expert-overlay-domain");

        if (!errors.empty())
            return errors;

        config.domain_definitions.clear();
        config.domain_definitions.reserve(inventory.size());
        for (const auto &domain : inventory)
            config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain));

        std::vector<ExecutionDomainDefinition> normalized_dense_domains;
        for (const auto &name : overlayDenseDomainNames(plan))
        {
            auto it = index_by_name.find(name);
            if (it == index_by_name.end())
                continue;
            normalized_dense_domains.push_back(inventory[it->second]);
        }

        if (!normalized_dense_domains.empty() || !plan.dense_domains.empty())
            plan.dense_domains = std::move(normalized_dense_domains);

        std::vector<ExpertComputeDomain> normalized_plan_domains;
        for (const auto &name : overlayRoutedDomainNames(plan))
        {
            auto it = index_by_name.find(name);
            if (it == index_by_name.end())
                continue;
            try
            {
                normalized_plan_domains.push_back(
                    ExpertComputeDomain::fromExecutionDomainDefinition(inventory[it->second]));
            }
            catch (const std::exception &e)
            {
                errors.push_back("MoE expert overlay domain '" + name + "': " + e.what());
            }
        }

        if (!normalized_plan_domains.empty() || !plan.domains.empty())
            plan.domains = std::move(normalized_plan_domains);

        return errors;
    }

    std::vector<std::string> validateMoEExpertOverlayConfig(const OrchestrationConfig &config)
    {
        std::vector<std::string> errors;
        const auto &plan_ptr = config.moe_expert_parallel_plan;
        if (!plan_ptr)
        {
            return errors;
        }

        const auto &plan = *plan_ptr;
        const bool has_overlay_details =
            !plan.continuation_domain.empty() ||
            !plan.base_model_domain.empty() ||
            !plan.shared_expert_domain.empty() ||
            plan.residency_policy != ExpertResidencyPolicy::Disabled ||
            !plan.continuation_domain_spec.domain.empty() ||
            !plan.dense_domains.empty() ||
            !plan.domains.empty() ||
            !plan.routed_tiers.empty() ||
            !plan.placements.empty();

        if (!plan.enabled)
        {
            if (has_overlay_details)
            {
                errors.push_back("MoE expert overlay is off/disabled but overlay domain, tier, continuation, shared-domain, residency, or placement settings were also provided");
            }
            return errors;
        }

        MoEExpertParallelValidationOptions validation_options;
        auto plan_result = validateMoEExpertParallelPlan(plan, validation_options);
        for (const auto &error : plan_result.errors)
        {
            errors.push_back("MoE expert overlay: " + error);
        }

        if (config.device_for_this_rank.has_value() && !plan.continuation_domain.empty())
        {
            const std::string device_spec = config.cpu_global_tp_all_local
                                                ? "cpu"
                                                : config.device_for_this_rank->toShortString();
            errors.push_back("Conflicting options: --device/-d " + device_spec +
                             " and --moe-expert-overlay-continuation " +
                             plan.continuation_domain +
                             ". Overlay continuation is the root/base placement; remove -d or disable overlay");
        }

        if (config.device_for_this_rank.has_value() && !plan.base_model_domain.empty())
        {
            const std::string device_spec = config.cpu_global_tp_all_local
                                                ? "cpu"
                                                : config.device_for_this_rank->toShortString();
            errors.push_back("Conflicting options: --device/-d " + device_spec +
                             " and --moe-expert-overlay-base-domain " +
                             plan.base_model_domain +
                             ". Overlay base/non-expert placement is explicit; remove -d or disable overlay");
        }

        auto domainByName = [&](const std::string &name) -> const ExpertComputeDomain *
        {
            auto it = std::find_if(plan.domains.begin(), plan.domains.end(),
                                   [&](const auto &domain)
                                   {
                                       return domain.name == name;
                                   });
            return it == plan.domains.end() ? nullptr : &*it;
        };

        const auto *continuation_domain = domainByName(plan.continuation_domain);
        const int continuation_owner = continuation_domain ? continuation_domain->owner_rank : -1;

        const std::string base_domain_name = plan.effectiveBaseModelDomain();
        const auto *base_domain = domainByName(base_domain_name);
        if (!plan.base_model_domain.empty() && base_domain && continuation_owner >= 0)
        {
            const int base_owner = base_domain->owner_rank;
            if (base_owner >= 0 && base_owner != continuation_owner)
            {
                errors.push_back("MoE expert overlay base/non-expert model domain '" + base_domain_name +
                                 "' owner rank " + std::to_string(base_owner) +
                                 " does not match continuation root rank " +
                                 std::to_string(continuation_owner) +
                                 "; current overlay root execution requires base and continuation placement on the same root rank");
            }
        }

        for (const auto &tier : plan.routed_tiers)
        {
            if (tier.domain == plan.continuation_domain)
                continue;

            const auto *domain = domainByName(tier.domain);
            if (!domain)
                continue;

            const bool remote_single_device =
                domain->kind == ExpertDomainKind::SingleDevice &&
                domain->owner_rank >= 0 &&
                continuation_owner >= 0 &&
                domain->owner_rank != continuation_owner;
            if (remote_single_device)
            {
                errors.push_back("MoE expert overlay auxiliary domain '" + domain->name +
                                 "' has no Phase 6 worker implementation for remote single-device replicated experts");
            }
        }

        if (!config.pp_stage_definitions.empty())
        {
            errors.push_back("MoE expert overlay cannot be combined with --pp-stage in Phase 2: overlay domains are same-layer expert roles, not PP layer ownership");
        }

        return errors;
    }

    std::vector<std::string> OrchestrationConfig::validate() const
    {
        std::vector<std::string> errors;

        // =====================================================================
        // Device Selection Validation (declarative rule framework)
        // =====================================================================
        {
            auto device_errors = ConfigValidator::createStandard().validateToStrings(*this);
            errors.insert(errors.end(), device_errors.begin(), device_errors.end());
        }

        // =====================================================================
        // Structural Validation (domain/PP/precision checks)
        // =====================================================================

        // Validate topology tree if present
        if (topology_tree)
        {
            auto tree_errors = topology_tree->validate();
            for (const auto &e : tree_errors)
            {
                errors.push_back("topology: " + e);
            }
        }

        // Validate domain definitions
        std::unordered_set<std::string> domain_names;
        for (const auto &domain : domain_definitions)
        {
            auto domain_errors = domain.validate();
            for (const auto &e : domain_errors)
            {
                errors.push_back(e);
            }

            if (domain_names.count(domain.name))
            {
                errors.push_back("Duplicate domain name: '" + domain.name + "'");
            }
            domain_names.insert(domain.name);
        }

        // Validate PP stage definitions
        std::set<int> stage_ids;
        for (const auto &stage : pp_stage_definitions)
        {
            auto stage_errors = stage.validate();
            for (const auto &e : stage_errors)
            {
                errors.push_back(e);
            }

            if (stage_ids.count(stage.stage_id))
            {
                errors.push_back("Duplicate PP stage ID: " + std::to_string(stage.stage_id));
            }
            stage_ids.insert(stage.stage_id);

            // Check domain reference exists
            if (!domain_names.empty() && !domain_names.count(stage.domain_name))
            {
                errors.push_back("PP stage " + std::to_string(stage.stage_id) +
                                 " references undefined domain '" + stage.domain_name + "'");
            }
        }

        // Validate TP configuration
        if (tp_degree < 1)
        {
            errors.push_back("TP degree must be >= 1, got " + std::to_string(tp_degree));
        }

        if (!tp_devices.empty() && !tp_weights.empty())
        {
            if (tp_devices.size() != tp_weights.size())
            {
                errors.push_back("TP devices (" + std::to_string(tp_devices.size()) +
                                 ") and weights (" + std::to_string(tp_weights.size()) +
                                 ") count mismatch");
            }

            float sum = std::accumulate(tp_weights.begin(), tp_weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
            {
                errors.push_back("TP weights sum to " + std::to_string(sum) + " (expected 1.0)");
            }
        }

        // Validate PP configuration
        if (pp_degree < 1)
        {
            errors.push_back("PP degree must be >= 1, got " + std::to_string(pp_degree));
        }

        // Validate device map
        std::set<int> mapped_ranks;
        for (const auto &[rank, addr] : device_map)
        {
            if (rank < 0)
            {
                errors.push_back("Device map has negative rank: " + std::to_string(rank));
            }
            if (mapped_ranks.count(rank))
            {
                errors.push_back("Device map has duplicate rank: " + std::to_string(rank));
            }
            mapped_ranks.insert(rank);
        }

        // Validate CPU layers
        if (cpu_layers < 0)
        {
            errors.push_back("CPU layers must be >= 0, got " + std::to_string(cpu_layers));
        }

        // Validate optional same-layer MoE expert overlay plan.
        {
            OrchestrationConfig normalized = *this;
            if (normalized.moe_expert_parallel_plan)
            {
                normalized.moe_expert_parallel_plan =
                    std::make_shared<MoEExpertParallelPlan>(*normalized.moe_expert_parallel_plan);
            }
            auto normalize_errors = normalizeMoEExpertOverlayDomains(normalized);
            for (const auto &error : normalize_errors)
            {
                errors.push_back("MoE expert overlay: " + error);
            }
            auto overlay_errors = validateMoEExpertOverlayConfig(normalized);
            errors.insert(errors.end(), overlay_errors.begin(), overlay_errors.end());
        }

        if (moe_hot_expert_cache.kind == MoEHotExpertCacheConfig::Kind::Count &&
            moe_hot_expert_cache.count < 0)
        {
            errors.push_back("MoE hot expert cache count must be >= 0");
        }
        if (moe_hot_expert_cache.kind == MoEHotExpertCacheConfig::Kind::Percent &&
            (moe_hot_expert_cache.percent < 0.0f || moe_hot_expert_cache.percent > 100.0f))
        {
            errors.push_back("MoE hot expert cache percent must be in [0, 100]");
        }
        if (moe_rebalance.window_size <= 0)
        {
            errors.push_back("MoE rebalance window must be > 0");
        }
        if (moe_rebalance.max_window_size < 0)
        {
            errors.push_back("MoE rebalance max window must be >= 0");
        }
        if (moe_rebalance.window_growth_factor <= 0.0f)
        {
            errors.push_back("MoE rebalance window growth factor must be > 0");
        }

        // Validate precision strings
        {
            const std::string act = toLower(activation_precision);
            static const std::unordered_set<std::string> valid_activation = {
                "fp32", "bf16", "fp16", "q8_1", "q16_1", "hybrid", "hybridq16"};
            if (!valid_activation.count(act))
            {
                errors.push_back("Invalid activation_precision: '" + activation_precision +
                                 "' (valid: fp32, bf16, fp16, q8_1, q16_1, hybrid, hybridq16)");
            }
        }

        {
            const std::string kv = toLower(kv_cache_precision);
            static const std::unordered_set<std::string> valid_kv = {
                "auto", "fp32", "f32", "fp16", "f16", "q8_1", "q8", "q81",
                "q16_1", "q16", "q161", "i16", "int16", "tq4", "tq"};
            if (!valid_kv.count(kv))
            {
                errors.push_back("Invalid kv_cache_precision: '" + kv_cache_precision +
                                 "' (valid: auto, fp32, fp16, q8_1, q16_1, tq4, tq)");
            }
        }

        return errors;
    }

    std::string OrchestrationConfig::toString() const
    {
        std::ostringstream oss;
        oss << "OrchestrationConfig {\n";

        // Introspection flags
        if (dry_run)
            oss << "  dry_run: true\n";
        if (explain_placement)
            oss << "  explain_placement: true\n";
        if (show_topology)
            oss << "  show_topology: true\n";
        if (show_numa)
            oss << "  show_numa: true\n";
        if (validate_only)
            oss << "  validate_only: true\n";

        // Device assignment
        oss << "  device_mode: " << deviceAssignmentModeToString(device_mode) << "\n";
        if (device_for_this_rank)
        {
            oss << "  device_for_this_rank: " << device_for_this_rank->toString() << "\n";
        }
        if (!device_map.empty())
        {
            oss << "  device_map: [";
            for (size_t i = 0; i < device_map.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << device_map[i].first << "=" << device_map[i].second.toShortString();
            }
            oss << "]\n";
        }

        // Named domains
        // Unified domain inventory
        const auto inventory = executionDomainDefinitions();
        if (!inventory.empty())
        {
            oss << "  execution_domains:\n";
            for (const auto &d : inventory)
            {
                oss << "    - " << d.toString() << "\n";
            }
        }
        if (!pp_stage_definitions.empty())
        {
            oss << "  placements:\n";
            oss << "    pp_layers:\n";
            for (const auto &s : pp_stage_definitions)
            {
                oss << "      - " << s.toString() << "\n";
            }
        }

        if (moe_expert_parallel_plan && moe_expert_parallel_plan->enabled)
        {
            const auto &plan = *moe_expert_parallel_plan;
            oss << renderMoEExpertParallelPlanExplanation(plan);
        }

        oss << "  moe:\n";
        oss << "    expert_mode: " << moeExpertModeToString(moe_expert_mode) << "\n";
        oss << "    rebalance: " << moeRebalanceRuntimeModeToString(moe_rebalance.mode) << "\n";
        oss << "    hot_expert_cache: " << moe_hot_expert_cache.toString() << "\n";
        oss << "    rebalance_window: " << moe_rebalance.window_size << "\n";
        oss << "    rebalance_max_window: " << moe_rebalance.max_window_size << "\n";
        oss << "    rebalance_window_growth: " << moe_rebalance.window_growth_factor << "\n";
        oss << "    release_raw_expert_weights: "
            << (moe_rebalance.release_raw_expert_weights ? "true" : "false") << "\n";

        // Simple TP
        oss << "  tp_degree: " << tp_degree << "\n";
        oss << "  tp_scope: " << tpScopeToString(tp_scope) << "\n";
        if (!tp_devices.empty())
        {
            oss << "  tp_devices: [";
            for (size_t i = 0; i < tp_devices.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << tp_devices[i].toShortString();
            }
            oss << "]\n";
        }

        // Simple PP
        oss << "  pp_degree: " << pp_degree << "\n";
        oss << "  pp_split: " << ppSplitModeToString(pp_split) << "\n";

        // Layer placement
        if (cpu_layers > 0)
        {
            oss << "  cpu_layers: " << cpu_layers;
            oss << " (" << (cpu_layers_first ? "first" : "last") << ")\n";
        }

        // Backend
        oss << "  default_backend: " << collectiveBackendTypeToString(default_backend) << "\n";

        // MPI bootstrap
        oss << "  mpi_profile: " << mpiProfileToString(mpi_profile) << "\n";

        // Config file
        if (!config_file_path.empty())
        {
            oss << "  config_file: " << config_file_path << "\n";
        }

        oss << "}";
        return oss.str();
    }

    OrchestrationConfig OrchestrationConfig::defaults()
    {
        return OrchestrationConfig{};
    }

} // namespace llaminar2
