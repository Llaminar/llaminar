/**
 * @file OrchestrationPlanningPolicy.h
 * @brief Typed automatic-selection intent shared by planning and serving.
 *
 * Parsing records user intent; resolution produces either a validated automatic
 * request or an apply-only request. Constraints restrict compute participants,
 * never the CPU control process or its physical-memory BOM. Preferences cannot
 * admit a forbidden participant or convert applying a saved plan into search.
 */
#pragma once

#include "backends/DeviceType.h"
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace llaminar2
{
    struct OrchestrationConfig;

    /** @brief Requested admission action before explicit placement is inspected. */
    enum class OrchestrationPlanningMode
    {
        InferFromPlacement, ///< No placement means auto; declared/saved placement means apply.
        Automatic, ///< Search within the hard constraints; reject an explicit placement.
        Apply, ///< Validate and apply the declared placement without choosing another one.
    };

    /** @brief Execution strategies, independent of backend/vendor and tier names. */
    enum class OrchestrationStrategy
    {
        SingleDevice,
        TensorParallel,
        PipelineParallel, ///< Includes a pipeline whose stages contain TP domains.
        ExpertOverlay,
    };

    /** @brief Hard physical-host participation intent, independent of MPI rank numbering. */
    enum class AutomaticHostParticipation
    {
        BestSubset, ///< Choose the lowest-cost admitted subset of observed hosts.
        AllDiscovered, ///< Every discovered physical host must supply a compute participant.
    };

    /**
     * @brief Expected request geometry used to rank automatic execution plans.
     *
     * This is a workload hint, not an inference limit or a measurement. Both
     * phases remain positive so a decode-only or prefill-only score cannot
     * silently replace the complete-request objective.
     */
    class OrchestrationPlanningWorkload final
    {
    public:
        /** @brief Seal positive request lengths; reject a missing phase. */
        OrchestrationPlanningWorkload(int prompt_tokens, int generation_tokens);
        /** @return Balanced default horizon bounded by the available context. */
        static OrchestrationPlanningWorkload defaultsForContext(int context_tokens);
        /** @return Number of newly prefilled tokens, not KV allocation capacity. */
        int promptTokens() const noexcept { return prompt_tokens_; }
        /** @return Number of generated tokens priced after the prompt. */
        int generationTokens() const noexcept { return generation_tokens_; }
        /** @brief Reject a horizon exceeding context without overflowing token arithmetic. */
        void requireFitsContext(int context_tokens) const;
        /** @brief Exact workload identity accompanies every cost observation. */
        bool operator==(const OrchestrationPlanningWorkload &) const = default;
    private:
        int prompt_tokens_;
        int generation_tokens_;
    };

    /**
     * @brief Ordered discovery ranks selected for one execution communicator.
     *
     * Configuration rank references use the compact execution namespace, while
     * this value alone records the enclosing discovery namespace. It contains
     * no hardware observation, allocation grant or process-lifetime authority.
     * Actual inventory bounds and collective agreement are checked before split.
     */
    class ExecutionRankSelection final
    {
    public:
        /** @brief Reject empty, negative or duplicate ranks without sorting away their order. */
        explicit ExecutionRankSelection(std::vector<int> discovery_ranks);
        /** @return Identity selection for one positive observed communicator size. */
        static ExecutionRankSelection all(int world_size);
        /** @return Discovery rank at each compact execution index. */
        const std::vector<int> &discoveryRanks() const noexcept { return ranks_; }
        /** @return Selected size, already checked to fit the MPI integer namespace. */
        int size() const noexcept { return static_cast<int>(ranks_.size()); }
        /** @return Compact execution index or absence; negative input is invalid. */
        std::optional<int> executionRank(int discovery_rank) const;
        /** @brief Compare exact ordered membership, not merely the selected set. */
        bool operator==(const ExecutionRankSelection &) const = default;

    private:
        std::vector<int> ranks_;
    };

    /**
     * @brief Exact physical compute-endpoint count for one backend in an auto plan.
     *
     * This restricts cardinality, not device identities, rank ownership, layer
     * boundaries or tier roles. CPU endpoints are observed NUMA participants,
     * not control processes or worker threads. The request validates positivity
     * and uniqueness before any inventory search or allocation.
     */
    struct AutomaticBackendDeviceCount
    {
        DeviceType backend;
        int count;
        bool operator==(const AutomaticBackendDeviceCount &) const = default;
    };

    /**
     * @brief Unresolved CLI/YAML options, validated once at the admission boundary.
     *
     * An omitted set permits all installed choices; an explicitly empty set is
     * an error. Keeping omission typed avoids treating an invalid filter as an
     * instruction to use every backend. Preferences carry no fabricated cost.
     */
    struct AutomaticOrchestrationOptions
    {
        std::optional<std::vector<DeviceType>> only_backends;
        std::optional<std::vector<OrchestrationStrategy>> only_strategies;
        std::optional<DeviceType> prefer_backend;
        std::optional<OrchestrationStrategy> prefer_strategy;
        std::optional<AutomaticHostParticipation> host_participation;
        std::optional<OrchestrationPlanningWorkload> workload;
        /** Unlisted backends remain governed by only_backends, not implicitly excluded. */
        std::optional<std::vector<AutomaticBackendDeviceCount>> device_counts;

        /** @return Whether any hard constraint or soft hint was explicitly supplied. */
        [[nodiscard]] bool specified() const noexcept;
    };

    /** @brief Validated constraints; construction cannot produce conflicting hints. */
    class AutomaticOrchestrationRequest final
    {
    public:
        /**
         * @brief Seal parsed options, rejecting empty sets and excluded preferences.
         * @param options User-supplied restrictions and hints; copied into the request.
         * @throws std::invalid_argument for unsupported values or contradictory intent.
         */
        explicit AutomaticOrchestrationRequest(AutomaticOrchestrationOptions options = {});

        /** @return Whether this compute backend is permitted, not whether it is available. */
        [[nodiscard]] bool allows(DeviceType backend) const noexcept;
        /** @return Whether this strategy is permitted, not whether its graph is installed. */
        [[nodiscard]] bool allows(OrchestrationStrategy strategy) const noexcept;
        /**
         * @brief Apply hard filters to every compute participant in a candidate.
         * @param strategy Candidate execution strategy.
         * @param participants One entry per distinct physical compute endpoint;
         *        excludes host control/BOM owners and repeated rank visibility.
         * @return False for empty membership, forbidden choices or unmet counts.
         */
        [[nodiscard]] bool allows(
            OrchestrationStrategy strategy, std::span<const DeviceType> participants) const noexcept;
        /** @return Immutable validated intent for ranking and explanation. */
        [[nodiscard]] const AutomaticOrchestrationOptions &options() const noexcept { return options_; }

    private:
        AutomaticOrchestrationOptions options_;
    };

    /** @brief Applying placement has no automatic candidate-search policy. */
    struct ApplyOrchestrationRequest {};
    using ResolvedOrchestrationIntent =
        std::variant<AutomaticOrchestrationRequest, ApplyOrchestrationRequest>;

    /** @return Stable public strategy spelling, throwing for an invalid enum. */
    std::string_view orchestrationStrategyName(OrchestrationStrategy strategy);
    /** @return Parsed installed strategy family; unknown or legacy aliases are errors. */
    OrchestrationStrategy parseOrchestrationStrategy(std::string_view value);
    /** @return Explicit automatic physical-host policy; unknown spellings are errors. */
    AutomaticHostParticipation parseAutomaticHostParticipation(std::string_view value);
    /** @return Exactly two positive integer token counts in prefill,generation order. */
    OrchestrationPlanningWorkload parseOrchestrationPlanningWorkload(std::string_view value);
    /** @return CPU/CUDA/ROCm compute backend; unsupported vendors are errors. */
    DeviceType parseOrchestrationComputeBackend(std::string_view value);
    /** @return Nonempty, duplicate-free comma-separated hard backend restriction. */
    std::vector<DeviceType> parseOrchestrationBackendList(std::string_view value);
    /** @return Nonempty, duplicate-free comma-separated hard strategy restriction. */
    std::vector<OrchestrationStrategy> parseOrchestrationStrategyList(std::string_view value);
    /** @return Exact positive backend=count pairs; duplicate backends are errors. */
    std::vector<AutomaticBackendDeviceCount> parseAutomaticDeviceCounts(std::string_view value);
    /**
     * @brief Resolve search versus apply once from explicit configuration intent.
     * @param config Shared runtime configuration after CLI/YAML merge.
     * @return Exactly one validated action, independent of command name or hardware.
     * @throws std::invalid_argument if auto conflicts with placement or apply lacks it.
     */
    ResolvedOrchestrationIntent resolveOrchestrationIntent(const OrchestrationConfig &config);
}
