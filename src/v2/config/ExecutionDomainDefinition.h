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

#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{

    enum class ExecutionDomainScope
    {
        AUTO,
        SINGLE,
        LOCAL,
        NODE_LOCAL,
        GLOBAL,
    };

    enum class ExecutionDomainComputeKind
    {
        UNSPECIFIED,
        REPLICATED_EXPERTS,
        EXPERT_ID_SHARDED,
        TENSOR_PARALLEL_EXPERTS,
    };

    const char *executionDomainScopeToString(ExecutionDomainScope scope);
    const char *executionDomainComputeKindToString(ExecutionDomainComputeKind kind);
    std::optional<ExecutionDomainScope> parseExecutionDomainScope(const std::string &value);
    std::optional<ExecutionDomainComputeKind> parseExecutionDomainComputeKind(const std::string &value);

    struct ExecutionDomainParseOptions
    {
        std::string context = "execution domain";
        bool require_scope = false;
        bool allow_single_scope = true;
        bool allow_global_scope = true;
        bool require_compute = false;
        bool allow_compute = true;
        bool allow_weights = true;
    };

    struct ExecutionDomainDefinition
    {
        std::string name;
        std::vector<GlobalDeviceAddress> participants;
        std::vector<float> weights;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        ExecutionDomainScope scope = ExecutionDomainScope::AUTO;
        std::optional<int> owner_rank;
        std::vector<int> ranks;
        ExecutionDomainComputeKind compute_kind = ExecutionDomainComputeKind::UNSPECIFIED;

        static ExecutionDomainDefinition parse(
            const std::string &spec,
            const ExecutionDomainParseOptions &options = {});

        static std::optional<ExecutionDomainDefinition> tryParse(
            const std::string &spec,
            const ExecutionDomainParseOptions &options = {});

        bool hasWeights() const { return !weights.empty(); }
        bool hasComputeKind() const { return compute_kind != ExecutionDomainComputeKind::UNSPECIFIED; }
        bool hasMultipleParticipants() const { return participants.size() > 1; }
        bool isDomainScopedTP() const;
        bool supportsTensorParallelExperts() const;
        bool supportsExpertIdSharding() const;

        std::string logicalIdentity() const { return name; }
        bool samePhysicalParticipants(const ExecutionDomainDefinition &other) const;

        std::vector<std::string> validate() const;
        std::string toString() const;
    };

} // namespace llaminar2