/**
 * @file ConfigValidator.h
 * @brief Declarative validation rule framework for OrchestrationConfig
 *
 * Provides a strategic, extensible approach to CLI option validation:
 *
 * 1. **DeviceSelectionMode**: Auto-detected from which CLI flags the user set.
 *    Mutually exclusive modes (SINGLE_DEVICE, EXPLICIT_TP, NAMED_DOMAINS, etc.)
 *    are enforced — conflicting flags produce clear error messages.
 *
 * 2. **ConfigValidationRule**: Each rule is a declarative struct with:
 *    - id: unique identifier (e.g., "device-tp-devices-mutex")
 *    - description: human-readable explanation
 *    - fix_hint: how to fix the violation
 *    - applies: predicate — does this rule fire?
 *    - check: returns error message if violated, nullopt if OK
 *
 * 3. **ConfigValidator**: Holds all rules, runs them against a config,
 *    returns structured errors. Created via `createStandard()` factory
 *    which defines all rules in one place.
 *
 * Usage:
 *   auto errors = ConfigValidator::createStandard().validate(config);
 *   // errors is a vector<ConfigValidationError> with id, message, fix_hint
 *
 * Adding new rules:
 *   Add a new addRule() call in ConfigValidator::createStandard() in
 *   ConfigValidator.cpp. That's it — no other files need to change.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declaration
    struct OrchestrationConfig;

    // =========================================================================
    // DeviceSelectionMode
    // =========================================================================

    /**
     * @brief Detected device selection mode based on which CLI flags were used.
     *
     * These modes are mutually exclusive. If flags from multiple modes are
     * detected, ConfigValidator will produce an error.
     *
     * | Mode            | Trigger Flags                                |
     * |-----------------|----------------------------------------------|
     * | UNSPECIFIED     | No device flags set → auto-detect             |
     * | SINGLE_DEVICE   | -d <device> only                             |
     * | DEVICE_MAP      | --device-map "0=cuda:0,1=cuda:1"             |
     * | SIMPLE_TP       | -tp N (auto-pick devices)                    |
     * | EXPLICIT_TP     | --tp-devices "cuda:0,cuda:1"                 |
     * | NAMED_DOMAINS   | --define-domain / --pp-stage                 |
     * | TOPOLOGY_TREE   | --topology / --topology-file                 |
     */
    enum class DeviceSelectionMode
    {
        UNSPECIFIED,   ///< No device selection flags — auto-detect
        SINGLE_DEVICE, ///< -d <device> (single device, no TP)
        DEVICE_MAP,    ///< --device-map (explicit rank→device mapping)
        SIMPLE_TP,     ///< -tp N without --tp-devices (auto-pick)
        EXPLICIT_TP,   ///< --tp-devices (exact device list for TP)
        NAMED_DOMAINS, ///< --define-domain and/or --pp-stage
        TOPOLOGY_TREE  ///< --topology or --topology-file
    };

    /**
     * @brief Convert DeviceSelectionMode to string for logging/diagnostics
     */
    const char *deviceSelectionModeToString(DeviceSelectionMode mode);

    /**
     * @brief Detect the device selection mode from a config.
     *
     * Examines which device-related fields are populated and returns the
     * detected mode. Does NOT validate — use ConfigValidator for that.
     *
     * If multiple modes are detected (conflict), returns the highest-priority
     * mode. ConfigValidator will separately flag the conflict as an error.
     *
     * @param config The config to analyze
     * @return Detected mode
     */
    DeviceSelectionMode detectDeviceSelectionMode(const OrchestrationConfig &config);

    // =========================================================================
    // ConfigValidationError
    // =========================================================================

    /**
     * @brief Structured validation error with context for user-friendly reporting
     */
    struct ConfigValidationError
    {
        std::string rule_id;  ///< Rule that was violated (e.g., "device-tp-devices-mutex")
        std::string message;  ///< Human-readable error message
        std::string fix_hint; ///< Suggested fix

        /**
         * @brief Format as a single-line error string
         */
        std::string toString() const;

        /**
         * @brief Format as a multi-line diagnostic block
         */
        std::string toDetailedString() const;
    };

    // =========================================================================
    // ConfigValidationRule
    // =========================================================================

    /**
     * @brief A single declarative validation rule.
     *
     * Rules are self-contained: they know when they apply, what to check,
     * and how to describe the fix. This makes them independently testable
     * and keeps all validation logic in one place (ConfigValidator::createStandard()).
     */
    struct ConfigValidationRule
    {
        /// Unique identifier (e.g., "device-tp-devices-mutex")
        std::string id;

        /// Human-readable description (e.g., "-d and --tp-devices are mutually exclusive")
        std::string description;

        /// How to fix the violation (shown to user)
        std::string fix_hint;

        /// Predicate: does this rule apply to this config?
        /// Return true if the rule should be checked.
        std::function<bool(const OrchestrationConfig &)> applies;

        /// Check: returns error message if violated, nullopt if the config is valid.
        /// Only called when applies() returns true.
        std::function<std::optional<std::string>(const OrchestrationConfig &)> check;
    };

    // =========================================================================
    // ConfigValidator
    // =========================================================================

    /**
     * @brief Declarative config validator with a table of rules.
     *
     * All device selection validation rules are defined in createStandard().
     * To add a new rule, add a single addRule() call there.
     *
     * The validator runs all applicable rules and collects all errors
     * (it doesn't stop at the first violation).
     */
    class ConfigValidator
    {
    public:
        /**
         * @brief Register a validation rule.
         */
        void addRule(ConfigValidationRule rule);

        /**
         * @brief Run all applicable rules against the config.
         * @param config The config to validate
         * @return List of validation errors (empty = valid)
         */
        std::vector<ConfigValidationError> validate(const OrchestrationConfig &config) const;

        /**
         * @brief Convenience: run validation and return plain error strings.
         *
         * Compatible with the existing OrchestrationConfig::validate() return type.
         */
        std::vector<std::string> validateToStrings(const OrchestrationConfig &config) const;

        /**
         * @brief Get the number of registered rules.
         */
        size_t ruleCount() const { return rules_.size(); }

        /**
         * @brief Get a rule by ID (for testing).
         * @return Pointer to the rule, or nullptr if not found.
         */
        const ConfigValidationRule *findRule(const std::string &id) const;

        /**
         * @brief Create a validator with all standard device selection rules.
         *
         * This is the single source of truth for all CLI option conflict rules.
         * Rules are organized by category:
         *
         * **Cross-Mode Mutual Exclusion:**
         *   - device-tp-devices-mutex: -d and --tp-devices can't coexist
         *   - device-named-domains-mutex: -d and --define-domain can't coexist
         *   - device-topology-mutex: -d and --topology can't coexist
         *   - tp-devices-named-domains-mutex: --tp-devices and --define-domain can't coexist
         *   - tp-devices-topology-mutex: --tp-devices and --topology can't coexist
         *   - named-domains-topology-mutex: --define-domain and --topology can't coexist
         *   - device-map-tp-devices-mutex: --device-map and --tp-devices can't coexist
         *   - device-map-named-domains-mutex: --device-map and --define-domain can't coexist
         *   - device-simple-tp-conflict: -d and -tp N (without --tp-devices) can't coexist
         *
         * **Co-Requirements:**
         *   - tp-weights-requires-tp-devices: --tp-weights requires --tp-devices
         *   - pp-stage-requires-domain: --pp-stage requires --define-domain
         *   - device-mode-explicit-requires-map: --device-mode explicit requires --device-map
         *
         * **Intra-Mode Consistency:**
         *   - tp-devices-degree-mismatch: if both --tp-devices and -tp are set, counts must match
         *   - tp-devices-no-duplicates: --tp-devices must not contain duplicates
         *   - tp-scope-global-tp-devices-conflict: --tp-scope global + --tp-devices is contradictory
         */
        static ConfigValidator createStandard();

    private:
        std::vector<ConfigValidationRule> rules_;
    };

} // namespace llaminar2
