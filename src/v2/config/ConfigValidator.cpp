/**
 * @file ConfigValidator.cpp
 * @brief Implementation of the declarative config validation framework
 *
 * All validation rules are defined in createStandard(). To add a new rule,
 * add a single addRule() call in that function. No other files need to change.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "config/ConfigValidator.h"
#include "config/OrchestrationConfig.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_set>

namespace llaminar2
{

    // =========================================================================
    // DeviceSelectionMode
    // =========================================================================

    const char *deviceSelectionModeToString(DeviceSelectionMode mode)
    {
        switch (mode)
        {
        case DeviceSelectionMode::UNSPECIFIED:
            return "unspecified (auto-detect)";
        case DeviceSelectionMode::SINGLE_DEVICE:
            return "single-device (-d)";
        case DeviceSelectionMode::DEVICE_MAP:
            return "device-map (--device-map)";
        case DeviceSelectionMode::SIMPLE_TP:
            return "simple-tp (-tp N, auto-pick)";
        case DeviceSelectionMode::EXPLICIT_TP:
            return "explicit-tp (--tp-devices)";
        case DeviceSelectionMode::NAMED_DOMAINS:
            return "named-domains (--define-domain)";
        case DeviceSelectionMode::TOPOLOGY_TREE:
            return "topology-tree (--topology)";
        default:
            return "unknown";
        }
    }

    // =========================================================================
    // Detection helpers (shared by detectDeviceSelectionMode and rules)
    // =========================================================================

    namespace
    {
        bool hasDevice(const OrchestrationConfig &c)
        {
            return c.device_for_this_rank.has_value();
        }

        bool hasDeviceMap(const OrchestrationConfig &c)
        {
            return !c.device_map.empty();
        }

        bool hasTPDevices(const OrchestrationConfig &c)
        {
            return !c.tp_devices.empty();
        }

        bool hasSimpleTP(const OrchestrationConfig &c)
        {
            // -tp N set (degree > 1) but no --tp-devices
            return c.tp_degree > 1 && c.tp_devices.empty();
        }

        bool hasNamedDomains(const OrchestrationConfig &c)
        {
            return !c.domain_definitions.empty() || !c.pp_stage_definitions.empty();
        }

        bool hasTopologyTree(const OrchestrationConfig &c)
        {
            return c.topology_tree.has_value() ||
                   !c.topology_string.empty() ||
                   !c.topology_file_path.empty();
        }

        bool hasTPWeights(const OrchestrationConfig &c)
        {
            return !c.tp_weights.empty();
        }

        bool hasPPStages(const OrchestrationConfig &c)
        {
            return !c.pp_stage_definitions.empty();
        }

    } // anonymous namespace

    DeviceSelectionMode detectDeviceSelectionMode(const OrchestrationConfig &config)
    {
        // Priority order (highest to lowest):
        // Named domains and topology tree are the most specific
        if (hasTopologyTree(config))
            return DeviceSelectionMode::TOPOLOGY_TREE;
        if (hasNamedDomains(config))
            return DeviceSelectionMode::NAMED_DOMAINS;
        if (hasTPDevices(config))
            return DeviceSelectionMode::EXPLICIT_TP;
        if (hasSimpleTP(config))
            return DeviceSelectionMode::SIMPLE_TP;
        if (hasDeviceMap(config))
            return DeviceSelectionMode::DEVICE_MAP;
        if (hasDevice(config))
            return DeviceSelectionMode::SINGLE_DEVICE;

        return DeviceSelectionMode::UNSPECIFIED;
    }

    // =========================================================================
    // ConfigValidationError
    // =========================================================================

    std::string ConfigValidationError::toString() const
    {
        std::ostringstream oss;
        oss << "[" << rule_id << "] " << message;
        if (!fix_hint.empty())
        {
            oss << " (fix: " << fix_hint << ")";
        }
        return oss.str();
    }

    std::string ConfigValidationError::toDetailedString() const
    {
        std::ostringstream oss;
        oss << "  Rule:    " << rule_id << "\n"
            << "  Error:   " << message << "\n";
        if (!fix_hint.empty())
        {
            oss << "  Fix:     " << fix_hint << "\n";
        }
        return oss.str();
    }

    // =========================================================================
    // ConfigValidator
    // =========================================================================

    void ConfigValidator::addRule(ConfigValidationRule rule)
    {
        rules_.push_back(std::move(rule));
    }

    std::vector<ConfigValidationError> ConfigValidator::validate(const OrchestrationConfig &config) const
    {
        std::vector<ConfigValidationError> errors;

        for (const auto &rule : rules_)
        {
            if (rule.applies && rule.applies(config))
            {
                if (rule.check)
                {
                    auto result = rule.check(config);
                    if (result.has_value())
                    {
                        errors.push_back({
                            .rule_id = rule.id,
                            .message = result.value(),
                            .fix_hint = rule.fix_hint,
                        });
                    }
                }
            }
        }

        return errors;
    }

    std::vector<std::string> ConfigValidator::validateToStrings(const OrchestrationConfig &config) const
    {
        auto errors = validate(config);
        std::vector<std::string> result;
        result.reserve(errors.size());
        for (const auto &e : errors)
        {
            result.push_back(e.toString());
        }
        return result;
    }

    const ConfigValidationRule *ConfigValidator::findRule(const std::string &id) const
    {
        for (const auto &rule : rules_)
        {
            if (rule.id == id)
            {
                return &rule;
            }
        }
        return nullptr;
    }

    // =========================================================================
    // createStandard() — THE SINGLE SOURCE OF TRUTH FOR ALL VALIDATION RULES
    // =========================================================================

    ConfigValidator ConfigValidator::createStandard()
    {
        ConfigValidator v;

        // =====================================================================
        // CROSS-MODE MUTUAL EXCLUSION RULES
        //
        // These prevent conflicting device selection mechanisms from being
        // used simultaneously. Each rule detects when two incompatible
        // flag groups are both populated.
        // =====================================================================

        v.addRule({
            .id = "device-tp-devices-mutex",
            .description = "--device (-d) and --tp-devices are mutually exclusive",
            .fix_hint = "Use --tp-devices to list all TP devices (it implies the primary device), "
                        "or use -d alone for single-device mode without tensor parallelism",
            .applies = [](const OrchestrationConfig &c)
            { return hasDevice(c) && hasTPDevices(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "Conflicting options: --device " +
                       c.device_for_this_rank->toShortString() +
                       " and --tp-devices both specified. "
                       "The primary device is determined by --tp-devices (first entry).";
            },
        });

        v.addRule({
            .id = "device-named-domains-mutex",
            .description = "--device (-d) and --define-domain are mutually exclusive",
            .fix_hint = "Named domains (--define-domain) fully control device assignment. "
                        "Remove -d and specify devices within domain definitions instead",
            .applies = [](const OrchestrationConfig &c)
            { return hasDevice(c) && hasNamedDomains(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "Conflicting options: --device " +
                       c.device_for_this_rank->toShortString() +
                       " and --define-domain both specified. "
                       "Named domains define their own device assignments.";
            },
        });

        v.addRule({
            .id = "device-topology-mutex",
            .description = "--device (-d) and --topology are mutually exclusive",
            .fix_hint = "The topology tree fully controls device placement. "
                        "Remove -d and specify devices within the topology definition",
            .applies = [](const OrchestrationConfig &c)
            { return hasDevice(c) && hasTopologyTree(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "Conflicting options: --device " +
                       c.device_for_this_rank->toShortString() +
                       " and --topology both specified. "
                       "The topology tree defines its own device assignments.";
            },
        });

        v.addRule({
            .id = "tp-devices-named-domains-mutex",
            .description = "--tp-devices and --define-domain are mutually exclusive",
            .fix_hint = "Use --define-domain for complex multi-domain setups (it includes TP device specification), "
                        "or use --tp-devices for simple homogeneous TP",
            .applies = [](const OrchestrationConfig &c)
            { return hasTPDevices(c) && hasNamedDomains(c); },
            .check = [](const OrchestrationConfig &) -> std::optional<std::string>
            {
                return "Conflicting options: --tp-devices and --define-domain both specified. "
                       "Named domains include their own device lists.";
            },
        });

        v.addRule({
            .id = "tp-devices-topology-mutex",
            .description = "--tp-devices and --topology are mutually exclusive",
            .fix_hint = "The topology tree defines TP device placement. "
                        "Remove --tp-devices and configure TP within the topology definition",
            .applies = [](const OrchestrationConfig &c)
            { return hasTPDevices(c) && hasTopologyTree(c); },
            .check = [](const OrchestrationConfig &) -> std::optional<std::string>
            {
                return "Conflicting options: --tp-devices and --topology both specified.";
            },
        });

        v.addRule({
            .id = "named-domains-topology-mutex",
            .description = "--define-domain and --topology are mutually exclusive",
            .fix_hint = "Use --topology for recursive parallelism trees, "
                        "or --define-domain + --pp-stage for flat multi-domain setups",
            .applies = [](const OrchestrationConfig &c)
            { return hasNamedDomains(c) && hasTopologyTree(c); },
            .check = [](const OrchestrationConfig &) -> std::optional<std::string>
            {
                return "Conflicting options: --define-domain and --topology both specified. "
                       "These are two different ways to define the same thing.";
            },
        });

        v.addRule({
            .id = "device-map-tp-devices-mutex",
            .description = "--device-map and --tp-devices are mutually exclusive",
            .fix_hint = "--device-map is for assigning one device per MPI rank; "
                        "--tp-devices is for multi-device TP within a rank. "
                        "Use one or the other, not both",
            .applies = [](const OrchestrationConfig &c)
            { return hasDeviceMap(c) && hasTPDevices(c); },
            .check = [](const OrchestrationConfig &) -> std::optional<std::string>
            {
                return "Conflicting options: --device-map and --tp-devices both specified.";
            },
        });

        v.addRule({
            .id = "device-map-named-domains-mutex",
            .description = "--device-map and --define-domain are mutually exclusive",
            .fix_hint = "Named domains fully control device assignment. "
                        "Remove --device-map and specify devices within domain definitions",
            .applies = [](const OrchestrationConfig &c)
            { return hasDeviceMap(c) && hasNamedDomains(c); },
            .check = [](const OrchestrationConfig &) -> std::optional<std::string>
            {
                return "Conflicting options: --device-map and --define-domain both specified.";
            },
        });

        v.addRule({
            .id = "device-simple-tp-conflict",
            .description = "--device (-d) and -tp N (without --tp-devices) conflict",
            .fix_hint = "For tensor parallelism, use --tp-devices to explicitly list all devices. "
                        "Example: --tp-devices \"cuda:0,cuda:1\" instead of -d cuda:0 -tp 2",
            .applies = [](const OrchestrationConfig &c)
            { return hasDevice(c) && hasSimpleTP(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "Conflicting options: --device " +
                       c.device_for_this_rank->toShortString() +
                       " and -tp " + std::to_string(c.tp_degree) +
                       " both specified. With -tp N, the system auto-picks N devices, "
                       "making -d ambiguous (is it a preference, constraint, or should be ignored?).";
            },
        });

        v.addRule({
            .id = "device-map-device-mutex",
            .description = "--device-map and --device (-d) are mutually exclusive",
            .fix_hint = "--device-map provides per-rank assignment; -d sets a single device. "
                        "Use one or the other",
            .applies = [](const OrchestrationConfig &c)
            { return hasDevice(c) && hasDeviceMap(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "Conflicting options: --device " +
                       c.device_for_this_rank->toShortString() +
                       " and --device-map both specified.";
            },
        });

        // =====================================================================
        // CO-REQUIREMENT RULES
        //
        // These ensure that dependent options are used together.
        // =====================================================================

        v.addRule({
            .id = "tp-weights-requires-tp-devices",
            .description = "--tp-weights requires --tp-devices",
            .fix_hint = "Specify --tp-devices to define which devices get the proportional weights. "
                        "Example: --tp-devices \"cuda:0,rocm:0\" --tp-weights \"0.73,0.27\"",
            .applies = [](const OrchestrationConfig &c)
            { return hasTPWeights(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                if (!hasTPDevices(c))
                {
                    return "--tp-weights specified without --tp-devices. "
                           "Weights need a device list to apply to.";
                }
                return std::nullopt;
            },
        });

        v.addRule({
            .id = "pp-stage-requires-domain",
            .description = "--pp-stage requires --define-domain",
            .fix_hint = "Define the domains that PP stages reference. "
                        "Example: --define-domain \"gpu_tp=cuda:0,cuda:1\" --pp-stage \"0=gpu_tp:0-13\"",
            .applies = [](const OrchestrationConfig &c)
            { return hasPPStages(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                if (c.domain_definitions.empty())
                {
                    return "--pp-stage specified without --define-domain. "
                           "PP stages must reference defined domains.";
                }
                return std::nullopt;
            },
        });

        v.addRule({
            .id = "device-mode-explicit-requires-map",
            .description = "--device-mode explicit requires --device-map",
            .fix_hint = "Provide a device map. "
                        "Example: --device-mode explicit --device-map \"0=cuda:0,1=cuda:1\"",
            .applies = [](const OrchestrationConfig &c)
            { return c.device_mode == DeviceAssignmentMode::EXPLICIT; },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                if (!hasDeviceMap(c))
                {
                    return "--device-mode explicit requires --device-map to specify rank-to-device assignments.";
                }
                return std::nullopt;
            },
        });

        // =====================================================================
        // INTRA-MODE CONSISTENCY RULES
        //
        // These validate that options within a single mode are consistent.
        // =====================================================================

        v.addRule({
            .id = "tp-devices-degree-mismatch",
            .description = "--tp-devices count must match -tp degree (if both set)",
            .fix_hint = "Either set only --tp-devices (degree is inferred from device count), "
                        "or ensure -tp N matches the number of devices in --tp-devices",
            .applies = [](const OrchestrationConfig &c)
            { return hasTPDevices(c) && c.tp_degree > 1; },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                if (static_cast<int>(c.tp_devices.size()) != c.tp_degree)
                {
                    return "--tp-devices has " + std::to_string(c.tp_devices.size()) +
                           " devices but -tp is set to " + std::to_string(c.tp_degree) +
                           ". These must match, or omit -tp (it will be inferred).";
                }
                return std::nullopt;
            },
        });

        v.addRule({
            .id = "tp-devices-no-duplicates",
            .description = "--tp-devices must not contain duplicate devices",
            .fix_hint = "Remove duplicate devices from --tp-devices. "
                        "Each TP shard must run on a distinct device",
            .applies = [](const OrchestrationConfig &c)
            { return hasTPDevices(c) && c.tp_devices.size() > 1; },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                std::set<GlobalDeviceAddress> seen;
                for (const auto &dev : c.tp_devices)
                {
                    if (!seen.insert(dev).second)
                    {
                        return "Duplicate device " + dev.toShortString() +
                               " in --tp-devices. Each tensor parallel shard "
                               "must run on a distinct device.";
                    }
                }
                return std::nullopt;
            },
        });

        v.addRule({
            .id = "tp-scope-global-tp-devices-conflict",
            .description = "--tp-scope global with --tp-devices is contradictory",
            .fix_hint = "--tp-scope global distributes TP across MPI ranks (one device per rank). "
                        "--tp-devices specifies local devices within a rank. "
                        "Use --tp-scope local with --tp-devices, or --tp-scope global without --tp-devices",
            .applies = [](const OrchestrationConfig &c)
            { return c.tp_scope == TPScope::GLOBAL && hasTPDevices(c); },
            .check = [](const OrchestrationConfig &) -> std::optional<std::string>
            {
                return "--tp-scope global uses MPI ranks for TP (one device per rank), "
                       "but --tp-devices specifies multiple local devices. "
                       "These modes are contradictory.";
            },
        });

        v.addRule({
            .id = "simple-tp-with-named-domains",
            .description = "-tp degree is ignored when using named domains",
            .fix_hint = "Named domains define their own TP degree via the device count in each domain. "
                        "Remove -tp N and configure TP within --define-domain",
            .applies = [](const OrchestrationConfig &c)
            { return c.tp_degree > 1 && hasNamedDomains(c) && !hasTPDevices(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "-tp " + std::to_string(c.tp_degree) +
                       " specified with --define-domain. Named domains control their own "
                       "TP degree via the number of devices in each domain definition.";
            },
        });

        v.addRule({
            .id = "simple-tp-with-topology",
            .description = "-tp degree is ignored when using topology tree",
            .fix_hint = "The topology tree defines its own parallelism structure. "
                        "Remove -tp N and configure TP within --topology",
            .applies = [](const OrchestrationConfig &c)
            { return c.tp_degree > 1 && hasTopologyTree(c) && !hasTPDevices(c); },
            .check = [](const OrchestrationConfig &c) -> std::optional<std::string>
            {
                return "-tp " + std::to_string(c.tp_degree) +
                       " specified with --topology. The topology tree defines "
                       "its own parallelism structure including TP.";
            },
        });

        return v;
    }

} // namespace llaminar2
