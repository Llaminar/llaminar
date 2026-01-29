/**
 * @file OrchestrationConfig.cpp
 * @brief Implementation of OrchestrationConfig and related structures
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "OrchestrationConfig.h"
#include "utils/Logger.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <cmath>
#include <set>
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

    const char *collectiveBackendTypeToString(CollectiveBackendType type)
    {
        switch (type)
        {
        case CollectiveBackendType::AUTO:
            return "auto";
        case CollectiveBackendType::NCCL:
            return "nccl";
        case CollectiveBackendType::RCCL:
            return "rccl";
        case CollectiveBackendType::PCIE_BAR:
            return "pcie_bar";
        case CollectiveBackendType::HETEROGENEOUS:
            return "heterogeneous";
        case CollectiveBackendType::UPI:
            return "upi";
        case CollectiveBackendType::MPI:
            return "mpi";
        case CollectiveBackendType::HOST:
            return "host";
        default:
            return "unknown";
        }
    }

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

    std::optional<CollectiveBackendType> parseCollectiveBackendType(const std::string &str)
    {
        std::string lower = toLower(str);
        if (lower == "auto")
            return CollectiveBackendType::AUTO;
        if (lower == "nccl")
            return CollectiveBackendType::NCCL;
        if (lower == "rccl")
            return CollectiveBackendType::RCCL;
        if (lower == "pciebar" || lower == "pcie_bar" || lower == "pcie-bar" || lower == "bar")
            return CollectiveBackendType::PCIE_BAR;
        if (lower == "heterogeneous" || lower == "hetero")
            return CollectiveBackendType::HETEROGENEOUS;
        if (lower == "upi")
            return CollectiveBackendType::UPI;
        if (lower == "mpi")
            return CollectiveBackendType::MPI;
        if (lower == "host")
            return CollectiveBackendType::HOST;
        return std::nullopt;
    }

    // =========================================================================
    // DomainDefinition Implementation
    // =========================================================================

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
        if (spec.empty())
        {
            return std::nullopt;
        }

        DomainDefinition def;

        // Split by ';' for different sections
        auto sections = split(spec, ';');
        if (sections.empty())
        {
            return std::nullopt;
        }

        // First section must be "name=devices"
        auto &main_section = sections[0];
        size_t eq_pos = main_section.find('=');
        if (eq_pos == std::string::npos)
        {
            return std::nullopt;
        }

        def.name = main_section.substr(0, eq_pos);
        if (def.name.empty())
        {
            return std::nullopt;
        }

        // Parse devices (comma-separated)
        std::string devices_str = main_section.substr(eq_pos + 1);
        auto device_specs = split(devices_str, ',');
        if (device_specs.empty())
        {
            return std::nullopt;
        }

        for (const auto &dev_spec : device_specs)
        {
            auto addr = GlobalDeviceAddress::tryParse(dev_spec);
            if (!addr)
            {
                return std::nullopt;
            }
            def.devices.push_back(*addr);
        }

        // Parse optional sections (weights, backend)
        for (size_t i = 1; i < sections.size(); ++i)
        {
            auto &section = sections[i];
            eq_pos = section.find('=');
            if (eq_pos == std::string::npos)
            {
                continue; // Ignore malformed sections
            }

            std::string key = toLower(section.substr(0, eq_pos));
            std::string value = section.substr(eq_pos + 1);

            if (key == "weights")
            {
                auto weight_strs = split(value, ',');
                for (const auto &w_str : weight_strs)
                {
                    try
                    {
                        float w = std::stof(w_str);
                        def.weights.push_back(w);
                    }
                    catch (const std::exception &)
                    {
                        return std::nullopt;
                    }
                }
            }
            else if (key == "backend")
            {
                auto backend = parseCollectiveBackendType(value);
                if (!backend)
                {
                    return std::nullopt;
                }
                def.backend = *backend;
            }
        }

        return def;
    }

    std::vector<std::string> DomainDefinition::validate() const
    {
        std::vector<std::string> errors;

        if (name.empty())
        {
            errors.push_back("Domain name cannot be empty");
        }

        if (devices.empty())
        {
            errors.push_back("Domain '" + name + "' has no devices");
        }

        if (!weights.empty())
        {
            if (weights.size() != devices.size())
            {
                errors.push_back("Domain '" + name + "' has " +
                                 std::to_string(devices.size()) + " devices but " +
                                 std::to_string(weights.size()) + " weights");
            }

            float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
            if (std::abs(sum - 1.0f) > 0.01f)
            {
                errors.push_back("Domain '" + name + "' weights sum to " +
                                 std::to_string(sum) + " (expected 1.0)");
            }

            for (size_t i = 0; i < weights.size(); ++i)
            {
                if (weights[i] < 0.0f || weights[i] > 1.0f)
                {
                    errors.push_back("Domain '" + name + "' has invalid weight " +
                                     std::to_string(weights[i]) + " (expected 0.0-1.0)");
                }
            }
        }

        return errors;
    }

    std::string DomainDefinition::toString() const
    {
        std::ostringstream oss;
        oss << name << "=[";
        for (size_t i = 0; i < devices.size(); ++i)
        {
            if (i > 0)
                oss << ",";
            oss << devices[i].toShortString();
        }
        oss << "]";

        if (!weights.empty())
        {
            oss << " weights=[";
            for (size_t i = 0; i < weights.size(); ++i)
            {
                if (i > 0)
                    oss << ",";
                oss << weights[i];
            }
            oss << "]";
        }

        if (backend != CollectiveBackendType::AUTO)
        {
            oss << " backend=" << collectiveBackendTypeToString(backend);
        }

        return oss.str();
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

    std::vector<std::string> OrchestrationConfig::validate() const
    {
        std::vector<std::string> errors;

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
        if (!domain_definitions.empty())
        {
            oss << "  domains:\n";
            for (const auto &d : domain_definitions)
            {
                oss << "    - " << d.toString() << "\n";
            }
        }
        if (!pp_stage_definitions.empty())
        {
            oss << "  pp_stages:\n";
            for (const auto &s : pp_stage_definitions)
            {
                oss << "    - " << s.toString() << "\n";
            }
        }

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
