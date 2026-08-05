/**
 * @file ExecutionDomainDefinition.h
 * @brief Canonical execution-domain contract shared by orchestration modes.
 *
 * Execution domains describe hardware participants and domain-internal
 * collective/compute capabilities. Role-specific placement, such as PP layer
 * ranges or MoE routed tiers, is intentionally kept outside this type.
 */

#pragma once

#include "backends/GlobalDeviceAddress.h"
#include "config/CollectiveBackendType.h"
#include "execution/config/RoutedExpertPolicy.h"

#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @enum ExecutionDomainScope
     * @brief Hardware reachability represented by an execution domain.
     *
     * Scope describes where participants live and therefore which collective
     * construction rules apply. It deliberately carries no dense or routed-
     * expert compute semantics.
     */
    enum class ExecutionDomainScope
    {
        AUTO,
        SINGLE,
        LOCAL,
        NODE_LOCAL,
        GLOBAL,
    };

    const char *executionDomainScopeToString(ExecutionDomainScope scope);
    std::optional<ExecutionDomainScope> parseExecutionDomainScope(const std::string &value);

    /**
     * @struct ExecutionDomainParseOptions
     * @brief Role-specific constraints applied by the shared domain parser.
     */
    struct ExecutionDomainParseOptions
    {
        ///< Human-readable owner used to contextualize parse errors.
        std::string context = "execution domain";
        ///< Require the declaration to state `scope=` explicitly.
        bool require_scope = false;
        ///< Permit `scope=single` for the consuming configuration surface.
        bool allow_single_scope = true;
        ///< Permit `scope=global` for the consuming configuration surface.
        bool allow_global_scope = true;
        ///< Require a `routed_compute=` policy in this domain declaration.
        bool require_routed_expert_compute = false;
        ///< Permit routed-expert policy fields on this configuration surface.
        bool allow_routed_expert_compute = true;
        ///< Permit proportional participant weights.
        bool allow_weights = true;
    };

    /**
     * @struct ExecutionDomainDefinition
     * @brief Canonical participant inventory shared by every orchestration mode.
     *
     * A domain combines physical participants, collective scope/backend, and
     * optional routed-expert capabilities. Model roles such as continuation,
     * pipeline stage, or routed tier remain in their respective placement plans.
     */
    struct ExecutionDomainDefinition
    {
        std::string name;
        std::vector<GlobalDeviceAddress> participants;
        std::vector<float> weights;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        ExecutionDomainScope scope = ExecutionDomainScope::AUTO;
        std::optional<int> owner_rank;
        std::vector<int> ranks;
        RoutedExpertComputePolicy routed_compute_policy =
            RoutedExpertComputePolicy::Unspecified;
        RoutedExpertPhasePolicy routed_phase_policy =
            RoutedExpertPhasePolicy::Unspecified;
        RoutedExpertAssignmentPolicy routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::Unspecified;
        RoutedExpertAssignmentPolicy routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::Unspecified;

        /**
         * @brief Parse a domain declaration or throw an actionable error.
         * @param spec Semicolon-delimited domain declaration.
         * @param options Constraints imposed by the caller's configuration role.
         * @return Fully parsed and validated canonical domain definition.
         * @throws std::invalid_argument if syntax or role constraints are invalid.
         */
        static ExecutionDomainDefinition parse(
            const std::string &spec,
            const ExecutionDomainParseOptions &options = {});

        /**
         * @brief Attempt to parse a domain declaration without propagating errors.
         * @param spec Semicolon-delimited domain declaration.
         * @param options Constraints imposed by the caller's configuration role.
         * @return Parsed domain when valid; otherwise `std::nullopt`.
         */
        static std::optional<ExecutionDomainDefinition> tryParse(
            const std::string &spec,
            const ExecutionDomainParseOptions &options = {});

        /** @brief Return true when proportional participant weights were supplied. */
        bool hasWeights() const { return !weights.empty(); }
        /** @brief Return true when routed-expert compute was stated explicitly. */
        bool hasRoutedExpertComputePolicy() const
        {
            return routed_compute_policy != RoutedExpertComputePolicy::Unspecified;
        }
        /** @brief Return true when phase-specific routed execution was stated explicitly. */
        bool hasRoutedExpertPhasePolicy() const
        {
            return routed_phase_policy != RoutedExpertPhasePolicy::Unspecified;
        }
        /** @brief Return true when decode row assignment was stated explicitly. */
        bool hasRoutedExpertDecodeAssignmentPolicy() const
        {
            return routed_decode_assignment_policy !=
                   RoutedExpertAssignmentPolicy::Unspecified;
        }
        /** @brief Return true when prefill row assignment was stated explicitly. */
        bool hasRoutedExpertPrefillAssignmentPolicy() const
        {
            return routed_prefill_assignment_policy !=
                   RoutedExpertAssignmentPolicy::Unspecified;
        }
        /** @brief Return true when the domain contains more than one participant. */
        bool hasMultipleParticipants() const { return participants.size() > 1; }

        /** @brief Return true when scope denotes an in-domain TP collective. */
        bool isDomainScopedTP() const;

        /** @brief Return true when complete expert IDs can be apportioned here. */
        bool supportsWholeExpertApportionment() const;

        /** @brief Return true when LLEP may schedule among complete residents. */
        bool supportsLeastLoadedResidentAssignment() const;

        /** @brief Return true when every expert may be tensor-sharded here. */
        bool supportsRoutedExpertTensorSharding() const;

        /** @brief Return the stable logical identity used by placement references. */
        std::string logicalIdentity() const { return name; }

        /**
         * @brief Compare physical participants while ignoring logical role names.
         * @param other Domain whose participant addresses are compared.
         * @return True when both domains describe the same physical participants.
         */
        bool samePhysicalParticipants(const ExecutionDomainDefinition &other) const;

        /** @brief Return every semantic validation error for this domain. */
        std::vector<std::string> validate() const;

        /** @brief Render the canonical domain declaration for diagnostics. */
        std::string toString() const;
    };

} // namespace llaminar2
