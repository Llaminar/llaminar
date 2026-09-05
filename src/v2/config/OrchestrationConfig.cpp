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
        case TPScope::RANK_LOCAL:
            return "rank_local";
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
        if (lower == "rank_local" || lower == "rank-local")
            return TPScope::RANK_LOCAL;
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
            case TPScope::RANK_LOCAL:
                return ExecutionDomainScope::RANK_LOCAL;
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
            case ExecutionDomainScope::RANK_LOCAL:
                return TPScope::RANK_LOCAL;
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
                   lhs.routed_compute_policy == rhs.routed_compute_policy &&
                   lhs.routed_phase_policy == rhs.routed_phase_policy &&
                   lhs.routed_decode_assignment_policy ==
                       rhs.routed_decode_assignment_policy &&
                   lhs.routed_prefill_assignment_policy ==
                       rhs.routed_prefill_assignment_policy;
        }

        bool participantSelectorAcceptsResolvedAddress(
            const GlobalDeviceAddress &selector,
            const GlobalDeviceAddress &resolved)
        {
            const bool wildcard_host = selector.hostname.empty() ||
                                       selector.hostname == "localhost";
            return selector.device_type == resolved.device_type &&
                   selector.device_ordinal == resolved.device_ordinal &&
                   (wildcard_host || selector.hostname == resolved.hostname) &&
                   (!selector.hasValidNuma() ||
                    selector.numa_node == resolved.numa_node);
        }

        bool sameDomainIntentAfterHardwareBinding(
            const ExecutionDomainDefinition &requested,
            const ExecutionDomainDefinition &resolved)
        {
            if (requested.name != resolved.name ||
                requested.participants.size() != resolved.participants.size() ||
                !sameWeights(requested.weights, resolved.weights) ||
                requested.backend != resolved.backend ||
                requested.routed_compute_policy != resolved.routed_compute_policy ||
                requested.routed_phase_policy != resolved.routed_phase_policy ||
                requested.routed_decode_assignment_policy !=
                    resolved.routed_decode_assignment_policy ||
                requested.routed_prefill_assignment_policy !=
                    resolved.routed_prefill_assignment_policy)
            {
                return false;
            }

            auto normalizedScope = [](const ExecutionDomainDefinition &domain)
            {
                if (domain.scope == ExecutionDomainScope::SINGLE &&
                    domain.participants.size() == 1)
                {
                    return ExecutionDomainScope::AUTO;
                }
                return domain.scope;
            };
            if (requested.scope != ExecutionDomainScope::AUTO &&
                normalizedScope(requested) != normalizedScope(resolved))
                return false;

            for (size_t index = 0; index < requested.participants.size(); ++index)
            {
                if (!participantSelectorAcceptsResolvedAddress(
                        requested.participants[index],
                        resolved.participants[index]))
                {
                    return false;
                }
            }

            if (requested.owner_rank.has_value() &&
                requested.owner_rank != resolved.owner_rank)
            {
                return false;
            }
            if (!requested.ranks.empty() && requested.ranks != resolved.ranks)
                return false;
            return true;
        }

        bool hasCompleteHardwareResolvedOwnership(
            const RoutedExpertDomain &domain)
        {
            if (domain.owner_rank < 0)
                return false;

            /*
             * A LocalTP domain is wholly owned by one MPI rank. Its devices do
             * not form a cross-rank participant list, so repeating the owner
             * once per device would violate the typed LOCAL-domain contract.
             */
            if (domain.scope == ExecutionDomainScope::RANK_LOCAL)
                return domain.world_ranks.empty();

            /*
             * SINGLE, NODE_LOCAL, and GLOBAL domains retain the binder's
             * participant-order rank map. NODE_LOCAL uses it to construct the
             * CPU collective; SINGLE uses its sole entry for remote ownership.
             */
            return domain.world_ranks.size() == domain.participants.size();
        }

        void addUniqueName(std::vector<std::string> &names, const std::string &name)
        {
            if (name.empty())
                return;
            if (std::find(names.begin(), names.end(), name) == names.end())
                names.push_back(name);
        }

        std::vector<std::string> routedPlacementDenseDomainNames(
            const MoERoutedExpertPlacementPlan &plan)
        {
            std::vector<std::string> names;
            addUniqueName(names, plan.continuation_domain);
            addUniqueName(names, plan.effectiveBaseModelDomain());
            addUniqueName(names, plan.shared_expert_domain);
            for (const auto &domain : plan.dense_domains)
                addUniqueName(names, domain.name);
            return names;
        }

        std::vector<std::string> routedPlacementDomainNames(
            const MoERoutedExpertPlacementPlan &plan)
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
        domain.routed_compute_policy = routed_compute_policy;
        domain.routed_phase_policy = routed_phase_policy;
        domain.routed_decode_assignment_policy =
            routed_decode_assignment_policy;
        domain.routed_prefill_assignment_policy =
            routed_prefill_assignment_policy;
        return domain;
    }

    DomainDefinition DomainDefinition::fromExecutionDomainDefinition(const ExecutionDomainDefinition &domain)
    {
        DomainDefinition def;
        def.name = domain.name;
        def.devices = domain.participants;
        def.weights = domain.weights;
        def.backend = domain.backend;
        def.routed_compute_policy = domain.routed_compute_policy;
        def.routed_phase_policy = domain.routed_phase_policy;
        def.routed_decode_assignment_policy =
            domain.routed_decode_assignment_policy;
        def.routed_prefill_assignment_policy =
            domain.routed_prefill_assignment_policy;
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
                        (moe_routed_expert_plan ? moe_routed_expert_plan->dense_domains.size() +
                                                        moe_routed_expert_plan->domains.size()
                                                  : 0));

        std::unordered_map<std::string, size_t> index_by_name;
        auto appendIfNew = [&](const ExecutionDomainDefinition &domain)
        {
            if (index_by_name.find(domain.name) != index_by_name.end())
                return;
            index_by_name.emplace(domain.name, domains.size());
            domains.push_back(domain);
        };

        if (moe_routed_expert_plan)
        {
            for (const auto &domain : moe_routed_expert_plan->dense_domains)
                appendIfNew(domain);
            for (const auto &domain : moe_routed_expert_plan->domains)
                appendIfNew(domain.toExecutionDomainDefinition());
        }

        for (const auto &domain : domain_definitions)
            appendIfNew(domain.toExecutionDomainDefinition());

        return domains;
    }

    std::vector<std::string> normalizeMoERoutedExpertPlacementDomains(
        OrchestrationConfig &config)
    {
        std::vector<std::string> errors;
        auto &plan_ptr = config.moe_routed_expert_plan;
        if (!plan_ptr)
            return errors;

        auto &plan = *plan_ptr;

        /*
         * Integer priority already defines a total preference order. Derive
         * final coverage from its least-preferred member so configuration
         * cannot carry a second, contradictory fallback ordering. Duplicate
         * priorities remain invalid and are diagnosed by plan validation;
         * selecting either duplicate here does not make that input admissible.
         */
        if (!plan.routed_tiers.empty())
        {
            auto coverage = std::max_element(
                plan.routed_tiers.begin(),
                plan.routed_tiers.end(),
                [](const RoutedExpertTier &lhs, const RoutedExpertTier &rhs)
                {
                    return lhs.priority < rhs.priority;
                });
            for (auto &tier : plan.routed_tiers)
                tier.fallback = (&tier == &*coverage);
        }

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
                                     "--moe-routed-expert-domain is a strict domain declaration and must not "
                                     "redefine a domain with different devices, scope, backend, ranks, weights, or routed-expert policies");
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
            addDomain(domain.toExecutionDomainDefinition(), "--moe-routed-expert-domain");

        if (!errors.empty())
            return errors;

        config.domain_definitions.clear();
        config.domain_definitions.reserve(inventory.size());
        for (const auto &domain : inventory)
            config.domain_definitions.push_back(DomainDefinition::fromExecutionDomainDefinition(domain));

        std::vector<ExecutionDomainDefinition> normalized_dense_domains;
        for (const auto &name : routedPlacementDenseDomainNames(plan))
        {
            auto it = index_by_name.find(name);
            if (it == index_by_name.end())
                continue;
            normalized_dense_domains.push_back(inventory[it->second]);
        }

        if (!normalized_dense_domains.empty() || !plan.dense_domains.empty())
            plan.dense_domains = std::move(normalized_dense_domains);

        std::vector<RoutedExpertDomain> normalized_plan_domains;
        for (const auto &name : routedPlacementDomainNames(plan))
        {
            auto it = index_by_name.find(name);
            if (it == index_by_name.end())
                continue;
            try
            {
                normalized_plan_domains.push_back(
                    RoutedExpertDomain::fromExecutionDomainDefinition(inventory[it->second]));
            }
            catch (const std::exception &e)
            {
                errors.push_back("MoE routed-expert domain '" + name + "': " + e.what());
            }
        }

        if (!normalized_plan_domains.empty() || !plan.domains.empty())
            plan.domains = std::move(normalized_plan_domains);

        return errors;
    }

    std::vector<std::string> installResolvedMoEExpertOverlayPlan(
        OrchestrationConfig &config,
        std::shared_ptr<MoERoutedExpertPlacementPlan> resolved_plan,
        MoEExpertOverlayPlanInstallOrigin origin)
    {
        std::vector<std::string> errors;
        if (!resolved_plan || !resolved_plan->usesExpertOverlayAuthority())
        {
            errors.push_back(
                "Hardware-resolved MoE overlay installation requires an enabled tiered plan");
            return errors;
        }

        if (origin ==
            MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP)
        {
            const bool has_simple_tp_authority =
                !config.tp_devices.empty() || config.tp_degree > 1;
            if (!has_simple_tp_authority)
            {
                errors.push_back(
                    "Implicit ExpertOverlay installation requires an active simple TP selector to replace");
            }
            if (!config.domain_definitions.empty() ||
                !config.pp_stage_definitions.empty())
            {
                errors.push_back(
                    "Implicit ExpertOverlay installation cannot replace simple TP while user-declared named domains remain active");
            }
            if (resolved_plan->topology !=
                    RoutedExpertPlacementTopology::SingleDomain ||
                resolved_plan->domains.size() != 1u ||
                resolved_plan->routed_tiers.size() != 1u ||
                resolved_plan->routed_tiers.front().priority != 0)
            {
                errors.push_back(
                    "Implicit ExpertOverlay installation requires one priority-zero single-domain plan");
            }
            if (!errors.empty())
                return errors;
        }
        if (!config.moe_routed_expert_plan ||
            !config.moe_routed_expert_plan->usesExpertOverlayAuthority())
        {
            errors.push_back(
                "Cannot install a hardware-resolved MoE overlay without a requested tiered plan");
            return errors;
        }

        std::unordered_map<std::string, const RoutedExpertDomain *> resolved_by_name;
        for (const auto &domain : resolved_plan->domains)
        {
            if (!resolved_by_name.emplace(domain.name, &domain).second)
            {
                errors.push_back(
                    "Hardware-resolved MoE overlay contains duplicate domain '" +
                    domain.name + "'");
            }
            if (!hasCompleteHardwareResolvedOwnership(domain))
            {
                errors.push_back(
                    "Hardware-resolved MoE overlay domain '" + domain.name +
                    "' does not have complete participant rank ownership");
            }
        }

        for (const auto &requested : config.moe_routed_expert_plan->domains)
        {
            auto found = resolved_by_name.find(requested.name);
            if (found == resolved_by_name.end())
            {
                errors.push_back(
                    "Hardware binding removed requested MoE overlay domain '" +
                    requested.name + "'");
                continue;
            }
            if (!sameDomainIntentAfterHardwareBinding(
                    requested.toExecutionDomainDefinition(),
                    found->second->toExecutionDomainDefinition()))
            {
                errors.push_back(
                    "Hardware binding changed non-location intent for MoE overlay domain '" +
                    requested.name + "'");
            }
        }

        std::vector<DomainDefinition> installed_domains =
            config.domain_definitions;
        for (const auto &resolved : resolved_plan->domains)
        {
            const auto resolved_definition =
                resolved.toExecutionDomainDefinition();
            auto existing = std::find_if(
                installed_domains.begin(),
                installed_domains.end(),
                [&](const auto &candidate)
                {
                    return candidate.name == resolved.name;
                });
            if (existing == installed_domains.end())
            {
                installed_domains.push_back(
                    DomainDefinition::fromExecutionDomainDefinition(
                        resolved_definition));
                continue;
            }

            if (!sameDomainIntentAfterHardwareBinding(
                    existing->toExecutionDomainDefinition(),
                    resolved_definition))
            {
                errors.push_back(
                    "Resolved MoE overlay domain '" + resolved.name +
                    "' conflicts with the canonical --define-domain hardware pool");
                continue;
            }
            *existing = DomainDefinition::fromExecutionDomainDefinition(
                resolved_definition);
        }

        if (!errors.empty())
            return errors;

        config.moe_routed_expert_plan = std::move(resolved_plan);
        config.domain_definitions = std::move(installed_domains);

        if (origin ==
            MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP)
        {
            /*
             * The installed named domain now contains the exact resolved
             * participants, weights, scope, and collective backend. Retiring
             * every simple-TP selector makes it the sole topology authority;
             * subsequent planners cannot accidentally choose the preliminary
             * representation or diagnose the internal translation as a user
             * conflict.
             */
            config.tp_devices.clear();
            config.tp_weights.clear();
            config.tp_degree = 1;
            config.tp_scope = TPScope::AUTO;
        }
        return errors;
    }

    std::vector<std::string> validateMoERoutedExpertPlacementConfig(
        const OrchestrationConfig &config)
    {
        std::vector<std::string> errors;
        const auto &plan_ptr = config.moe_routed_expert_plan;
        if (!plan_ptr)
        {
            return errors;
        }

        const auto &plan = *plan_ptr;
        const bool has_routed_placement_details =
            !plan.continuation_domain.empty() ||
            !plan.base_model_domain.empty() ||
            !plan.shared_expert_domain.empty() ||
            plan.residency_policy != RoutedExpertResidencyPolicy::Disabled ||
            !plan.continuation_domain_spec.domain.empty() ||
            plan.continuation_domain_spec.dense_tp_enabled ||
            !plan.dense_domains.empty() ||
            !plan.domains.empty() ||
            !plan.routed_tiers.empty() ||
            !plan.placements.empty();

        if (!plan.enabled)
        {
            if (has_routed_placement_details)
            {
                errors.push_back("MoE routed-expert placement is off but domain, tier, continuation, shared-domain, residency, or placement settings were also provided");
            }
            return errors;
        }

        MoERoutedExpertPlacementValidationOptions validation_options;
        auto plan_result = validateMoERoutedExpertPlacementPlan(plan, validation_options);
        for (const auto &error : plan_result.errors)
        {
            errors.push_back("MoE routed-expert placement: " + error);
        }

        if (config.device_for_this_rank.has_value() && !plan.continuation_domain.empty())
        {
            const std::string device_spec = config.cpu_global_tp_all_local
                                                ? "cpu"
                                                : config.device_for_this_rank->toShortString();
            errors.push_back("Conflicting options: --device/-d " + device_spec +
                             " and --moe-routed-expert-continuation-domain " +
                             plan.continuation_domain +
                             ". Routed-expert continuation placement is explicit; remove -d or disable routed-expert placement");
        }

        if (config.device_for_this_rank.has_value() && !plan.base_model_domain.empty())
        {
            const std::string device_spec = config.cpu_global_tp_all_local
                                                ? "cpu"
                                                : config.device_for_this_rank->toShortString();
            errors.push_back("Conflicting options: --device/-d " + device_spec +
                             " and --moe-routed-expert-base-model-domain " +
                             plan.base_model_domain +
                             ". Base/non-expert placement is explicit; remove -d or disable routed-expert placement");
        }

        auto domainByName = [&](const std::string &name) -> const RoutedExpertDomain *
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
                errors.push_back("MoE routed-expert placement base/non-expert model domain '" + base_domain_name +
                                 "' owner rank " + std::to_string(base_owner) +
                                 " does not match continuation root rank " +
                                 std::to_string(continuation_owner) +
                                 "; current routed-placement root execution requires base and continuation placement on the same root rank");
            }
        }

        if (!config.pp_stage_definitions.empty())
        {
            errors.push_back("MoE routed-expert placement cannot be combined with --pp-stage in Phase 2: routed domains are same-layer expert roles, not PP layer ownership");
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

        // Validate the optional same-layer routed-expert placement plan.
        {
            OrchestrationConfig normalized = *this;
            if (normalized.moe_routed_expert_plan)
            {
                normalized.moe_routed_expert_plan =
                    std::make_shared<MoERoutedExpertPlacementPlan>(*normalized.moe_routed_expert_plan);
            }
            auto normalize_errors = normalizeMoERoutedExpertPlacementDomains(normalized);
            for (const auto &error : normalize_errors)
            {
                errors.push_back("MoE routed-expert placement: " + error);
            }
            auto placement_errors = validateMoERoutedExpertPlacementConfig(normalized);
            errors.insert(errors.end(), placement_errors.begin(), placement_errors.end());
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
        if (moe_rebalance.migration_payoff_horizon_tokens == 0)
        {
            errors.push_back(
                "MoE migration payoff horizon tokens must be > 0");
        }
        if (moe_rebalance.migration_transfer_slots == 0)
        {
            errors.push_back(
                "MoE migration transfer slots must be > 0");
        }
        if (moe_rebalance.resolvedMigrationExecutionStreams() == 0)
        {
            errors.push_back(
                "MoE migration execution streams must be > 0");
        }
        if (moe_rebalance.resolvedMigrationExecutionStreams() >
            moe_rebalance.migration_transfer_slots)
        {
            errors.push_back(
                "MoE migration execution streams cannot exceed transfer slots");
        }
        if (moe_rebalance.resolvedMigrationCyclesPerWave() == 0)
        {
            errors.push_back(
                "MoE migration cycles per wave must be > 0");
        }
        if (moe_rebalance.resolvedMigrationCyclesPerWave() >
            moe_rebalance.migration_transfer_slots)
        {
            errors.push_back(
                "MoE migration cycles per wave cannot exceed physical transfer slots");
        }
        if (moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
        {
            /*
             * Dynamic ownership compares max/min load as a per-mille ratio.
             * Its mathematical floor is 1.0; reject an impossible threshold
             * at configuration admission instead of deferring the same error
             * until the residency authority has allocated its dependencies.
             */
            if (moe_rebalance.dynamic_imbalance_threshold_per_mille <
                moe_rebalance_policy::
                    kMinimumDynamicImbalanceThresholdPerMille)
            {
                errors.push_back(
                    "MoE Dynamic imbalance threshold must be >= 1000 per-mille");
            }
            if (moe_rebalance.dynamic_max_swaps_per_layer == 0u)
            {
                errors.push_back(
                    "MoE Dynamic maximum swaps per layer must be > 0");
            }
            if (moe_rebalance.dynamic_max_plan_entries_per_wave < 2u)
            {
                errors.push_back(
                    "MoE Dynamic plan entries per wave must admit at least one paired swap (>= 2)");
            }
        }
        if (moe_routed_prefill.assignment_window_tokens < 0)
        {
            errors.push_back("MoE routed-prefill assignment window tokens must be >= 0");
        }
        if (moe_routed_prefill.overlay_segment_rows <= 0)
        {
            errors.push_back("MoE overlay prefill segment rows must be > 0");
        }
        if (moe_routed_prefill.llep_alpha_numerator == 0)
        {
            errors.push_back("MoE routed-prefill LLEP alpha numerator must be > 0");
        }
        if (moe_routed_prefill.llep_alpha_denominator == 0)
        {
            errors.push_back("MoE routed-prefill LLEP alpha denominator must be > 0");
        }
        if (moe_routed_prefill.llep_lambda_numerator == 0)
        {
            errors.push_back("MoE routed-prefill LLEP lambda numerator must be > 0");
        }
        if (moe_routed_prefill.llep_lambda_denominator == 0)
        {
            errors.push_back("MoE routed-prefill LLEP lambda denominator must be > 0");
        }
        if (moe_rebalance.device_maintenance_slack_tokens < -1)
        {
            errors.push_back("MoE device maintenance slack tokens must be >= -1");
        }
        if (moe_rebalance.device_min_maintenance_period_tokens < -1)
        {
            errors.push_back("MoE device minimum maintenance period tokens must be >= -1");
        }
        if (moe_rebalance.device_initial_maintenance_period_tokens < -1)
        {
            errors.push_back("MoE device initial maintenance period tokens must be >= -1");
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

        if (!tp_allreduce_precision_override.empty())
        {
            const std::string tp_ar = toLower(tp_allreduce_precision_override);
            static const std::unordered_set<std::string> valid_tp_ar = {
                "auto", "schema", "default", "off", "fp32", "f32", "fp16", "f16", "bf16"};
            if (!valid_tp_ar.count(tp_ar))
            {
                errors.push_back("Invalid tp_allreduce_precision_override: '" +
                                 tp_allreduce_precision_override +
                                 "' (valid: auto, schema, fp32, fp16, bf16)");
            }
        }

        if (prefix_cache.block_size <= 0)
        {
            errors.push_back("Prefix cache block size must be > 0");
        }
        if (mtp.graph_capacity_draft_tokens < 0)
        {
            errors.push_back(
                "MTP graph capacity draft tokens must be >= 0");
        }
        if (mtp.enabled)
        {
            if (mtp.draft_tokens <= 0)
            {
                errors.push_back("MTP draft tokens must be > 0");
            }
            if (mtp.max_request_batch <= 0)
            {
                errors.push_back("MTP max request batch must be > 0");
            }
            const auto &depth_policy = mtp.depth_policy;
            if (depth_policy.min_depth < 0)
            {
                errors.push_back("MTP depth policy min depth must be >= 0");
            }
            if (depth_policy.max_depth < 0)
            {
                errors.push_back("MTP depth policy max depth must be >= 0");
            }
            if (depth_policy.initial_depth < 0)
            {
                errors.push_back("MTP depth policy initial depth must be >= 0");
            }

            if (depth_policy.mode != MTPDepthPolicyMode::Fixed)
            {
                /*
                 * Fixed-depth MTP is normalized by MTPDepthController to
                 * min=max=initial=draft_tokens and ignores dynamic hysteresis
                 * fields. Validate the rolling-window knobs only for observe
                 * and dynamic modes so hard-pinned benchmark lanes are not
                 * coupled to adaptive-policy defaults.
                 */
                const int effective_max_depth =
                    depth_policy.max_depth > 0 ? depth_policy.max_depth : mtp.draft_tokens;
                const int effective_initial_depth =
                    resolveMTPDepthPolicyInitialDepth(
                        depth_policy,
                        mtp.draft_tokens,
                        mtp.verify_mode);
                if (effective_max_depth < depth_policy.min_depth)
                {
                    errors.push_back("MTP depth policy max depth must be >= min depth");
                }
                if (effective_initial_depth < depth_policy.min_depth ||
                    effective_initial_depth > effective_max_depth)
                {
                    errors.push_back("MTP depth policy initial depth must be within [min depth, max depth]");
                }
                if (depth_policy.window_size <= 0)
                {
                    errors.push_back("MTP depth policy window size must be > 0");
                }
                if (depth_policy.min_samples <= 0)
                {
                    errors.push_back("MTP depth policy min samples must be > 0");
                }
                if (depth_policy.cooldown_steps < 0)
                {
                    errors.push_back("MTP depth policy cooldown steps must be >= 0");
                }
                if (depth_policy.promote_consecutive_windows <= 0)
                {
                    errors.push_back("MTP depth policy promote consecutive windows must be > 0");
                }
                auto valid_rate = [](double value)
                {
                    return value >= 0.0 && value <= 1.0;
                };
                if (!valid_rate(depth_policy.promote_full_accept_rate) ||
                    !valid_rate(depth_policy.demote_zero_accept_rate) ||
                    !valid_rate(depth_policy.demote_acceptance_rate))
                {
                    errors.push_back("MTP depth policy thresholds must be in [0, 1]");
                }
            }
            const int execution_maximum =
                resolveMTPMaximumExecutionDraftDepth(mtp);
            if (mtp.graph_capacity_draft_tokens > 0 &&
                mtp.graph_capacity_draft_tokens < execution_maximum)
            {
                errors.push_back(
                    "MTP graph capacity draft tokens must cover the maximum "
                    "execution-policy depth");
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

        if (moe_routed_expert_plan && moe_routed_expert_plan->enabled)
        {
            const auto &plan = *moe_routed_expert_plan;
            oss << renderMoERoutedExpertPlacementPlanExplanation(plan);
        }

        oss << "  moe:\n";
        oss << "    routed_expert_compute_policy: "
            << routedExpertComputePolicyToString(routed_expert_compute_policy) << "\n";
        oss << "    routed_expert_owner_order: "
            << routedExpertOwnerOrderToString(routed_expert_owner_order) << "\n";
        oss << "    residency_maintenance: "
            << moeRebalanceRuntimeModeToString(moe_rebalance.mode) << "\n";
        oss << "    hot_expert_cache: " << moe_hot_expert_cache.toString() << "\n";
        oss << "    residency_maintenance_window: "
            << moe_rebalance.window_size << "\n";
        oss << "    residency_maintenance_max_window: "
            << moe_rebalance.max_window_size << "\n";
        oss << "    residency_maintenance_window_growth: "
            << moe_rebalance.window_growth_factor << "\n";
        oss << "    migration_payoff_horizon_tokens: "
            << moe_rebalance.migration_payoff_horizon_tokens << "\n";
        oss << "    migration_transfer_slots: "
            << moe_rebalance.migration_transfer_slots << "\n";
        oss << "    migration_execution_streams: "
            << moe_rebalance.resolvedMigrationExecutionStreams() << "\n";
        oss << "    migration_cycles_per_wave: "
            << moe_rebalance.resolvedMigrationCyclesPerWave() << "\n";
        oss << "    routed_prefill_assignment_window_tokens: "
            << moe_routed_prefill.assignment_window_tokens << "\n";
        oss << "    overlay_prefill_segment_rows: "
            << moe_routed_prefill.overlay_segment_rows << "\n";
        oss << "    routed_prefill_least_loaded_min_routed_rows: "
            << moe_routed_prefill.least_loaded_min_routed_rows << "\n";
        oss << "    routed_prefill_llep_alpha: "
            << moe_routed_prefill.llep_alpha_numerator
            << "/" << moe_routed_prefill.llep_alpha_denominator << "\n";
        oss << "    routed_prefill_llep_lambda: "
            << moe_routed_prefill.llep_lambda_numerator
            << "/" << moe_routed_prefill.llep_lambda_denominator << "\n";
        oss << "    routed_prefill_llep_enable_balanced_skip: "
            << (moe_routed_prefill.llep_enable_balanced_skip ? "true" : "false") << "\n";
        oss << "    dynamic_imbalance_threshold_permille: "
            << moe_rebalance.dynamic_imbalance_threshold_per_mille << "\n";
        oss << "    dynamic_min_improvement_permille: "
            << moe_rebalance.dynamic_min_improvement_per_mille << "\n";
        oss << "    dynamic_max_swaps_per_layer: "
            << moe_rebalance.dynamic_max_swaps_per_layer << "\n";
        oss << "    dynamic_max_plan_entries_per_wave: "
            << moe_rebalance.dynamic_max_plan_entries_per_wave << "\n";
        oss << "    dynamic_min_window_activations: "
            << moe_rebalance.dynamic_min_window_activations << "\n";
        oss << "    device_rebalance_maintenance_slack_tokens: "
            << moe_rebalance.device_maintenance_slack_tokens << "\n";
        oss << "    device_rebalance_min_maintenance_period_tokens: "
            << moe_rebalance.device_min_maintenance_period_tokens << "\n";
        oss << "    device_rebalance_initial_maintenance_period_tokens: "
            << moe_rebalance.device_initial_maintenance_period_tokens << "\n";
        oss << "    device_min_load_spread_improvement: "
            << moe_rebalance.device_min_load_spread_improvement << "\n";
        oss << "    device_min_load_spread_improvement_divisor: "
            << moe_rebalance.device_min_load_spread_improvement_divisor << "\n";
        oss << "    device_min_wave_spread_improvement_per_payload_slot: "
            << moe_rebalance.device_min_wave_spread_improvement_per_payload_slot << "\n";
        oss << "    device_min_foreign_rows_per_critical_path_payload_slot: "
            << moe_rebalance
                   .device_min_foreign_rows_per_critical_path_payload_slot
            << "\n";
        oss << "    device_min_router_spread_improvement_per_payload_slot: "
            << moe_rebalance.device_min_router_spread_improvement_per_payload_slot << "\n";
        oss << "    device_max_post_wave_load_spread_permille: "
            << moe_rebalance.device_max_post_wave_load_spread_per_mille << "\n";
        oss << "    release_raw_expert_weights: "
            << (moe_rebalance.release_raw_expert_weights ? "true" : "false") << "\n";

        if (benchmark_mode || !benchmark_json_output_path.empty())
        {
            oss << "  benchmark:\n";
            oss << "    enabled: " << (benchmark_mode ? "true" : "false") << "\n";
            if (!benchmark_json_output_path.empty())
                oss << "    json_output: " << benchmark_json_output_path << "\n";
        }

        oss << "  precision:\n";
        oss << "    activation: " << activation_precision << "\n";
        oss << "    kv_cache: " << kv_cache_precision << "\n";
        if (!tp_allreduce_precision_override.empty())
            oss << "    tp_allreduce: " << tp_allreduce_precision_override << "\n";

        oss << "  prefix_cache:\n";
        oss << "    enabled: " << (prefix_cache.enabled ? "true" : "false") << "\n";
        oss << "    storage: " << prefixCacheStorageModeToString(prefix_cache.storage_mode) << "\n";
        oss << "    block_size: " << prefix_cache.block_size << "\n";
        oss << "    ram_budget_bytes: " << prefix_cache.ram_budget_bytes << "\n";
        oss << "    device_budget_bytes: " << prefix_cache.device_budget_bytes << "\n";
        oss << "    disk_budget_bytes: " << prefix_cache.disk_budget_bytes << "\n";
        if (!prefix_cache.disk_dir.empty())
            oss << "    disk_dir: " << prefix_cache.disk_dir << "\n";
        oss << "    terminal_state: " << prefixCacheTerminalStateModeToString(prefix_cache.terminal_state) << "\n";
        oss << "    moe_policy: " << prefixCacheMoEPolicyToString(prefix_cache.moe_policy) << "\n";

        oss << "  mtp:\n";
        oss << "    enabled: " << (mtp.enabled ? "true" : "false") << "\n";
        oss << "    draft_tokens: " << mtp.draft_tokens << "\n";
        oss << "    graph_capacity_draft_tokens: "
            << mtp.graph_capacity_draft_tokens << "\n";
        oss << "    max_request_batch: " << mtp.max_request_batch << "\n";
        oss << "    verify_mode: " << mtpVerifyModeToString(mtp.verify_mode) << "\n";
        oss << "    terminal_head_policy: "
            << mtpTerminalHeadPolicyToString(mtp.terminal_head_policy) << "\n";
        oss << "    depth_policy: " << mtpDepthPolicyModeToString(mtp.depth_policy.mode) << "\n";
        oss << "    min_draft_tokens: " << mtp.depth_policy.min_depth << "\n";
        oss << "    max_draft_tokens: " << mtp.depth_policy.max_depth << "\n";
        oss << "    initial_draft_tokens: " << mtp.depth_policy.initial_depth << "\n";
        oss << "    depth_window: " << mtp.depth_policy.window_size << "\n";
        oss << "    depth_min_samples: " << mtp.depth_policy.min_samples << "\n";
        oss << "    depth_cooldown: " << mtp.depth_policy.cooldown_steps << "\n";
        oss << "    depth_promote_windows: "
            << mtp.depth_policy.promote_consecutive_windows << "\n";
        oss << "    depth_generated_policy: "
            << (mtp.depth_policy.use_generated_policy ? "true" : "false") << "\n";
        oss << "    depth_promote_full_accept: " << mtp.depth_policy.promote_full_accept_rate << "\n";
        oss << "    depth_demote_zero_accept: " << mtp.depth_policy.demote_zero_accept_rate << "\n";
        oss << "    depth_demote_acceptance: " << mtp.depth_policy.demote_acceptance_rate << "\n";
        oss << "    require_terminal_hidden_for_full_hit: "
            << (mtp.require_terminal_hidden_for_full_hit ? "true" : "false") << "\n";

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
