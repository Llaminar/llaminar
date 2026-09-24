#include "config/ExecutionDomainDefinition.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        std::string trim(const std::string &str)
        {
            const size_t start = str.find_first_not_of(" \t\n\r");
            if (start == std::string::npos)
                return "";
            const size_t end = str.find_last_not_of(" \t\n\r");
            return str.substr(start, end - start + 1);
        }

        std::string toLower(const std::string &str)
        {
            std::string lower = str;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            return lower;
        }

        std::string normalizeToken(const std::string &str)
        {
            std::string normalized = toLower(trim(str));
            std::replace(normalized.begin(), normalized.end(), '-', '_');
            return normalized;
        }

        std::vector<std::string> split(const std::string &str, char delim)
        {
            std::vector<std::string> parts;
            std::stringstream ss(str);
            std::string part;
            while (std::getline(ss, part, delim))
            {
                std::string trimmed = trim(part);
                if (!trimmed.empty())
                    parts.push_back(std::move(trimmed));
            }
            return parts;
        }

        int parseNonNegativeInt(const std::string &value, const std::string &field_name)
        {
            try
            {
                const int parsed = std::stoi(value);
                if (parsed < 0)
                    throw std::invalid_argument("negative value");
                return parsed;
            }
            catch (const std::exception &)
            {
                throw std::invalid_argument("Invalid execution domain " + field_name + " value: '" + value + "'");
            }
        }

        std::vector<int> parseRankList(const std::string &value, const std::string &field_name)
        {
            const auto rank_parts = split(value, ',');
            if (rank_parts.empty())
                throw std::invalid_argument("Execution domain " + field_name + " must not be empty");

            std::vector<int> ranks;
            ranks.reserve(rank_parts.size());
            for (const auto &part : rank_parts)
                ranks.push_back(parseNonNegativeInt(part, field_name));
            return ranks;
        }
    } // namespace

    const char *executionDomainScopeToString(ExecutionDomainScope scope)
    {
        switch (scope)
        {
        case ExecutionDomainScope::AUTO:
            return "auto";
        case ExecutionDomainScope::SINGLE:
            return "single";
        case ExecutionDomainScope::RANK_LOCAL:
            return "rank_local";
        case ExecutionDomainScope::NODE_LOCAL:
            return "node_local";
        case ExecutionDomainScope::GLOBAL:
            return "global";
        }
        return "unknown";
    }

    std::optional<ExecutionDomainScope> parseExecutionDomainScope(const std::string &value)
    {
        const std::string normalized = normalizeToken(value);
        if (normalized == "auto")
            return ExecutionDomainScope::AUTO;
        if (normalized == "single" || normalized == "single_device")
            return ExecutionDomainScope::SINGLE;
        if (normalized == "rank_local")
            return ExecutionDomainScope::RANK_LOCAL;
        if (normalized == "node_local")
            return ExecutionDomainScope::NODE_LOCAL;
        if (normalized == "global")
            return ExecutionDomainScope::GLOBAL;
        return std::nullopt;
    }

    ExecutionDomainDefinition ExecutionDomainDefinition::parse(
        const std::string &spec,
        const ExecutionDomainParseOptions &options)
    {
        if (trim(spec).empty())
            throw std::invalid_argument(options.context + " spec is empty");

        const auto sections = split(spec, ';');
        if (sections.empty())
            throw std::invalid_argument(options.context + " spec is empty");

        const size_t eq_pos = sections[0].find('=');
        if (eq_pos == std::string::npos)
        {
            throw std::invalid_argument("Invalid " + options.context + " spec: '" + spec +
                                        "' (expected name=devices[;scope=...][;backend=...][;routed_compute=...][;routed_phase=...][;routed_decode_assignment=...][;routed_prefill_assignment=...])");
        }

        ExecutionDomainDefinition domain;
        domain.name = trim(sections[0].substr(0, eq_pos));
        if (domain.name.empty())
            throw std::invalid_argument(options.context + " name must not be empty");

        const auto device_specs = split(sections[0].substr(eq_pos + 1), ',');
        if (device_specs.empty())
            throw std::invalid_argument(options.context + " '" + domain.name + "' must declare at least one participant");

        domain.participants.reserve(device_specs.size());
        for (const auto &device_spec : device_specs)
        {
            auto addr = GlobalDeviceAddress::tryParse(device_spec);
            if (!addr)
                throw std::invalid_argument("Invalid " + options.context + " participant: '" + device_spec + "'");
            domain.participants.push_back(*addr);
        }

        bool saw_scope = false;
        bool saw_routed_compute = false;
        for (size_t index = 1; index < sections.size(); ++index)
        {
            const size_t option_eq_pos = sections[index].find('=');
            if (option_eq_pos == std::string::npos)
                throw std::invalid_argument("Invalid " + options.context + " option: '" + sections[index] + "'");

            const std::string key = normalizeToken(sections[index].substr(0, option_eq_pos));
            const std::string value = trim(sections[index].substr(option_eq_pos + 1));

            if (key == "weights")
            {
                if (!options.allow_weights)
                    throw std::invalid_argument(options.context + " does not accept weights");
                const auto weight_parts = split(value, ',');
                if (weight_parts.empty())
                    throw std::invalid_argument(options.context + " weights must not be empty");
                for (const auto &weight_part : weight_parts)
                {
                    try
                    {
                        domain.weights.push_back(std::stof(weight_part));
                    }
                    catch (const std::exception &)
                    {
                        throw std::invalid_argument("Invalid " + options.context + " weight: '" + weight_part + "'");
                    }
                }
            }
            else if (key == "backend")
            {
                auto backend = parseCollectiveBackendType(value);
                if (!backend)
                    throw std::invalid_argument("Invalid " + options.context + " backend: '" + value + "'");
                domain.backend = *backend;
            }
            else if (key == "scope")
            {
                auto scope = parseExecutionDomainScope(value);
                if (!scope)
                    throw std::invalid_argument("Invalid " + options.context + " scope: '" + value + "'");
                if (*scope == ExecutionDomainScope::SINGLE && !options.allow_single_scope)
                    throw std::invalid_argument(options.context + " scope=single is not valid for this mode");
                if (*scope == ExecutionDomainScope::GLOBAL && !options.allow_global_scope)
                    throw std::invalid_argument(options.context + " scope=global is not valid for this mode");
                domain.scope = *scope;
                saw_scope = true;
            }
            else if (key == "owner")
            {
                const auto owner_ranks = parseRankList(value, "owner");
                if (owner_ranks.size() != 1)
                    throw std::invalid_argument(options.context + " owner must be a single world rank");
                domain.owner_rank = owner_ranks.front();
            }
            else if (key == "ranks")
            {
                domain.ranks = parseRankList(value, "ranks");
            }
            else if (key == "routed_compute")
            {
                if (!options.allow_routed_expert_compute)
                    throw std::invalid_argument(options.context + " does not accept routed_compute=<policy>");
                auto compute = parseRoutedExpertComputePolicy(value);
                if (!compute)
                    throw std::invalid_argument("Invalid " + options.context + " routed compute policy: '" + value + "'");
                domain.routed_compute_policy = *compute;
                saw_routed_compute = true;
            }
            else if (key == "routed_decode_assignment")
            {
                auto assignment = parseRoutedExpertAssignmentPolicy(value);
                if (!assignment)
                    throw std::invalid_argument("Invalid " + options.context + " routed decode assignment policy: '" + value + "'");
                domain.routed_decode_assignment_policy = *assignment;
            }
            else if (key == "routed_prefill_assignment")
            {
                auto assignment = parseRoutedExpertAssignmentPolicy(value);
                if (!assignment)
                    throw std::invalid_argument("Invalid " + options.context + " routed prefill assignment policy: '" + value + "'");
                domain.routed_prefill_assignment_policy = *assignment;
            }
            else if (key == "routed_phase")
            {
                auto phase = parseRoutedExpertPhasePolicy(value);
                if (!phase)
                    throw std::invalid_argument("Invalid " + options.context + " routed phase policy: '" + value + "'");
                domain.routed_phase_policy = *phase;
            }
            else
            {
                throw std::invalid_argument("Unknown " + options.context + " option: '" + key + "'");
            }
        }

        if (options.require_scope && !saw_scope)
            throw std::invalid_argument(options.context + " '" + domain.name + "' is missing scope=<auto|single|rank_local|node_local>");
        if (options.require_routed_expert_compute && !saw_routed_compute)
            throw std::invalid_argument(options.context + " '" + domain.name + "' is missing routed_compute=<replicated|apportioned|tensor-sharded>");

        return domain;
    }

    std::optional<ExecutionDomainDefinition> ExecutionDomainDefinition::tryParse(
        const std::string &spec,
        const ExecutionDomainParseOptions &options)
    {
        try
        {
            return parse(spec, options);
        }
        catch (const std::exception &)
        {
            return std::nullopt;
        }
    }

    bool ExecutionDomainDefinition::isDomainScopedTP() const
    {
        return scope == ExecutionDomainScope::RANK_LOCAL || scope == ExecutionDomainScope::NODE_LOCAL;
    }

    bool ExecutionDomainDefinition::supportsWholeExpertApportionment() const
    {
        return !participants.empty();
    }

    bool ExecutionDomainDefinition::supportsLeastLoadedResidentAssignment() const
    {
        return isDomainScopedTP() && hasMultipleParticipants();
    }

    bool ExecutionDomainDefinition::supportsRoutedExpertTensorSharding() const
    {
        return isDomainScopedTP() && hasMultipleParticipants();
    }

    bool ExecutionDomainDefinition::samePhysicalParticipants(const ExecutionDomainDefinition &other) const
    {
        return participants == other.participants;
    }

    std::vector<std::string> ExecutionDomainDefinition::validate() const
    {
        std::vector<std::string> errors;

        if (name.empty())
            errors.push_back("Domain name cannot be empty");

        if (participants.empty())
            errors.push_back("Domain '" + name + "' has no devices");

        if (!weights.empty())
        {
            if (weights.size() != participants.size())
            {
                errors.push_back("Domain '" + name + "' has " +
                                 std::to_string(participants.size()) + " devices but " +
                                 std::to_string(weights.size()) + " weights");
            }

            const float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
                errors.push_back("Domain '" + name + "' weights sum to " + std::to_string(sum) + " (expected 1.0)");

            for (float weight : weights)
            {
                if (weight < 0.0f || weight > 1.0f)
                    errors.push_back("Domain '" + name + "' has invalid weight " + std::to_string(weight) + " (expected 0.0-1.0)");
            }
        }

        if (scope == ExecutionDomainScope::SINGLE && participants.size() > 1)
            errors.push_back("Domain '" + name + "' has scope=single but multiple participants specified");

        if (scope == ExecutionDomainScope::RANK_LOCAL && !ranks.empty())
            errors.push_back("Domain '" + name + "' has scope=rank_local but explicit_ranks is set (use owner= for rank-local domains)");

        if (owner_rank.has_value() && *owner_rank < 0)
            errors.push_back("Domain '" + name + "' has invalid owner_rank " + std::to_string(*owner_rank));

        std::set<int> participating_ranks;
        for (int rank : ranks)
        {
            if (rank < 0)
                errors.push_back("Domain '" + name + "' has a negative rank");
            else
                participating_ranks.insert(rank);
        }

        if (owner_rank.has_value() && !ranks.empty() &&
            participating_ranks.find(*owner_rank) ==
                participating_ranks.end())
        {
            errors.push_back("Domain '" + name + "' owner rank " + std::to_string(*owner_rank) +
                             " is not in its rank list");
        }

        if (routed_compute_policy == RoutedExpertComputePolicy::Apportioned &&
            !supportsWholeExpertApportionment())
        {
            errors.push_back("Domain '" + name + "' uses routed_compute=apportioned but has no participants");
        }

        const bool apportioned_prefill_over_replicated_weights =
            routed_compute_policy == RoutedExpertComputePolicy::Replicated &&
            routed_phase_policy ==
                RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated;

        if (routed_phase_policy ==
                RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated &&
            routed_compute_policy != RoutedExpertComputePolicy::Replicated)
        {
            errors.push_back(
                "Domain '" + name +
                "' uses routed_phase=prefill-apportioned-decode-replicated "
                "but routed_compute is not replicated");
        }

        if (routed_phase_policy ==
                RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated &&
            !supportsLeastLoadedResidentAssignment())
        {
            errors.push_back(
                "Domain '" + name +
                "' uses routed_phase=prefill-apportioned-decode-replicated "
                "but is not a multi-participant domain-scoped TP domain");
        }

        const auto validate_assignment =
            [&](RoutedExpertAssignmentPolicy assignment,
                const char *configuration_key,
                bool phase_allows_replicated_apportionment)
        {
            if (assignment != RoutedExpertAssignmentPolicy::LeastLoadedResident)
                return;

            if (routed_compute_policy != RoutedExpertComputePolicy::Apportioned &&
                !(phase_allows_replicated_apportionment &&
                  apportioned_prefill_over_replicated_weights))
            {
                errors.push_back(
                    "Domain '" + name + "' uses " + configuration_key +
                    "=least-loaded-resident but that workload does not use "
                    "participant-assigned complete experts");
            }

            if (!supportsLeastLoadedResidentAssignment())
            {
                errors.push_back(
                    "Domain '" + name + "' uses " + configuration_key +
                    "=least-loaded-resident but is not a multi-participant "
                    "domain-scoped TP domain");
            }
        };
        validate_assignment(
            routed_decode_assignment_policy,
            "routed_decode_assignment",
            false);
        validate_assignment(
            routed_prefill_assignment_policy,
            "routed_prefill_assignment",
            true);

        if (routed_compute_policy == RoutedExpertComputePolicy::TensorSharded &&
            !supportsRoutedExpertTensorSharding())
        {
            errors.push_back("Domain '" + name + "' uses routed_compute=tensor-sharded but is not a multi-participant domain-scoped TP domain");
        }

        return errors;
    }

    std::string ExecutionDomainDefinition::toString() const
    {
        std::ostringstream oss;
        oss << name << "=[";
        for (size_t index = 0; index < participants.size(); ++index)
        {
            if (index > 0)
                oss << ",";
            oss << participants[index].toShortString();
        }
        oss << "]";

        if (!weights.empty())
        {
            oss << " weights=[";
            for (size_t index = 0; index < weights.size(); ++index)
            {
                if (index > 0)
                    oss << ",";
                oss << weights[index];
            }
            oss << "]";
        }

        if (scope != ExecutionDomainScope::AUTO)
            oss << " scope=" << executionDomainScopeToString(scope);
        if (owner_rank.has_value())
            oss << " owner=" << *owner_rank;
        if (!ranks.empty())
        {
            oss << " ranks=[";
            for (size_t index = 0; index < ranks.size(); ++index)
            {
                if (index > 0)
                    oss << ",";
                oss << ranks[index];
            }
            oss << "]";
        }
        if (backend != CollectiveBackendType::AUTO)
            oss << " backend=" << collectiveBackendTypeToString(backend);
        if (routed_compute_policy != RoutedExpertComputePolicy::Unspecified)
            oss << " routed_compute="
                << routedExpertComputePolicyToString(routed_compute_policy);
        if (routed_phase_policy != RoutedExpertPhasePolicy::Unspecified)
            oss << " routed_phase="
                << routedExpertPhasePolicyToString(routed_phase_policy);
        if (routed_decode_assignment_policy !=
            RoutedExpertAssignmentPolicy::Unspecified)
        {
            oss << " routed_decode_assignment="
                << routedExpertAssignmentPolicyToString(
                       routed_decode_assignment_policy);
        }
        if (routed_prefill_assignment_policy !=
            RoutedExpertAssignmentPolicy::Unspecified)
        {
            oss << " routed_prefill_assignment="
                << routedExpertAssignmentPolicyToString(
                       routed_prefill_assignment_policy);
        }

        return oss.str();
    }

} // namespace llaminar2
