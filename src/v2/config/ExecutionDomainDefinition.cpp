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
        case ExecutionDomainScope::LOCAL:
            return "local";
        case ExecutionDomainScope::NODE_LOCAL:
            return "node_local";
        case ExecutionDomainScope::GLOBAL:
            return "global";
        }
        return "unknown";
    }

    const char *executionDomainComputeKindToString(ExecutionDomainComputeKind kind)
    {
        switch (kind)
        {
        case ExecutionDomainComputeKind::UNSPECIFIED:
            return "unspecified";
        case ExecutionDomainComputeKind::REPLICATED_EXPERTS:
            return "replicated_experts";
        case ExecutionDomainComputeKind::EXPERT_ID_SHARDED:
            return "expert_id_sharded";
        case ExecutionDomainComputeKind::TENSOR_PARALLEL_EXPERTS:
            return "tensor_parallel_experts";
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
        if (normalized == "local" || normalized == "local_tp")
            return ExecutionDomainScope::LOCAL;
        if (normalized == "node_local" || normalized == "nodelocal" || normalized == "node_local_tp")
            return ExecutionDomainScope::NODE_LOCAL;
        if (normalized == "global")
            return ExecutionDomainScope::GLOBAL;
        return std::nullopt;
    }

    std::optional<ExecutionDomainComputeKind> parseExecutionDomainComputeKind(const std::string &value)
    {
        const std::string normalized = normalizeToken(value);
        if (normalized == "replicated_experts")
            return ExecutionDomainComputeKind::REPLICATED_EXPERTS;
        if (normalized == "expert_id_sharded")
            return ExecutionDomainComputeKind::EXPERT_ID_SHARDED;
        if (normalized == "tensor_parallel_experts")
            return ExecutionDomainComputeKind::TENSOR_PARALLEL_EXPERTS;
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
                                        "' (expected name=devices[;scope=...][;backend=...][;compute=...])");
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
        bool saw_compute = false;
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
            else if (key == "compute")
            {
                if (!options.allow_compute)
                    throw std::invalid_argument(options.context + " does not accept compute=<kind>");
                auto compute = parseExecutionDomainComputeKind(value);
                if (!compute)
                    throw std::invalid_argument("Invalid " + options.context + " compute kind: '" + value + "'");
                domain.compute_kind = *compute;
                saw_compute = true;
            }
            else
            {
                throw std::invalid_argument("Unknown " + options.context + " option: '" + key + "'");
            }
        }

        if (options.require_scope && !saw_scope)
            throw std::invalid_argument(options.context + " '" + domain.name + "' is missing scope=<single|local|node_local>");
        if (options.require_compute && !saw_compute)
            throw std::invalid_argument(options.context + " '" + domain.name + "' is missing compute=<replicated_experts|expert_id_sharded|tensor_parallel_experts>");

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
        return scope == ExecutionDomainScope::LOCAL || scope == ExecutionDomainScope::NODE_LOCAL;
    }

    bool ExecutionDomainDefinition::supportsTensorParallelExperts() const
    {
        return isDomainScopedTP() && hasMultipleParticipants();
    }

    bool ExecutionDomainDefinition::supportsExpertIdSharding() const
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

        if (scope == ExecutionDomainScope::LOCAL && !ranks.empty())
            errors.push_back("Domain '" + name + "' has scope=local but explicit_ranks is set (use owner= for local domains)");

        if (owner_rank.has_value() && *owner_rank < 0)
            errors.push_back("Domain '" + name + "' has invalid owner_rank " + std::to_string(*owner_rank));

        std::set<int> seen_ranks;
        for (int rank : ranks)
        {
            if (rank < 0)
                errors.push_back("Domain '" + name + "' has a negative rank");
            else if (!seen_ranks.insert(rank).second)
                errors.push_back("Domain '" + name + "' has duplicate rank " + std::to_string(rank));
        }

        if (owner_rank.has_value() && !ranks.empty() && seen_ranks.find(*owner_rank) == seen_ranks.end())
        {
            errors.push_back("Domain '" + name + "' owner rank " + std::to_string(*owner_rank) +
                             " is not in its rank list");
        }

        if ((compute_kind == ExecutionDomainComputeKind::EXPERT_ID_SHARDED) && !supportsExpertIdSharding())
        {
            errors.push_back("Domain '" + name + "' uses expert_id_sharded but is not a multi-participant domain-scoped TP domain");
        }

        if ((compute_kind == ExecutionDomainComputeKind::TENSOR_PARALLEL_EXPERTS) && !supportsTensorParallelExperts())
        {
            errors.push_back("Domain '" + name + "' uses tensor_parallel_experts but is not a multi-participant domain-scoped TP domain");
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
        if (compute_kind != ExecutionDomainComputeKind::UNSPECIFIED)
            oss << " compute=" << executionDomainComputeKindToString(compute_kind);

        return oss.str();
    }

} // namespace llaminar2