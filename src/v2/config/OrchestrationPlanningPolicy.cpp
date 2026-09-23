/**
 * @file OrchestrationPlanningPolicy.cpp
 * @brief Fail-closed intent resolution and backend-symmetric candidate filters.
 *
 * This module performs no discovery, allocation, model loading or optimization.
 * It separates user constraints from installed-path eligibility and measured
 * costs, so neither a missing device nor an unfavorable estimate can relax a
 * hard constraint. Apply requests cannot contain an automatic search policy.
 * Ordered execution membership is a validated value shared by saved configs
 * and the inventory projector; MPI alone owns admission and communicator split.
 */
#include "config/OrchestrationPlanningPolicy.h"
#include "config/ConfigValidator.h"
#include "config/OrchestrationConfig.h"
#include <algorithm>
#include <charconv>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    OrchestrationPlanningWorkload::OrchestrationPlanningWorkload(int prompt_tokens, int generation_tokens)
        : prompt_tokens_(prompt_tokens), generation_tokens_(generation_tokens)
    {
        if (prompt_tokens <= 0 || generation_tokens <= 0)
            throw std::invalid_argument("Automatic planning needs positive prompt and generation lengths");
    }

    OrchestrationPlanningWorkload OrchestrationPlanningWorkload::defaultsForContext(int context_tokens)
    {
        if (context_tokens < 2)
            throw std::invalid_argument("Automatic orchestration requires context length >= 2");
        // Keep a meaningful generation horizon instead of pricing one token.
        // Small contexts share their capacity without eliminating either phase.
        constexpr int default_phase_tokens = 256;
        const int prompt = std::min(default_phase_tokens, context_tokens / 2);
        return {prompt, std::min(default_phase_tokens, context_tokens - prompt)};
    }

    void OrchestrationPlanningWorkload::requireFitsContext(int context_tokens) const
    {
        // Check before subtracting: even INT_MAX-sized user input cannot wrap.
        if (prompt_tokens_ > context_tokens || generation_tokens_ > context_tokens - prompt_tokens_)
            throw std::invalid_argument("Automatic planning workload exceeds the requested context capacity");
    }

    ExecutionRankSelection::ExecutionRankSelection(std::vector<int> discovery_ranks)
        : ranks_(std::move(discovery_ranks))
    {
        if (ranks_.empty() || ranks_.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Execution selection requires a nonempty MPI rank list");
        std::set<int> distinct;
        for (const int rank : ranks_)
            if (rank < 0 || !distinct.insert(rank).second)
                throw std::invalid_argument("Execution selection contains a negative or duplicate discovery rank");
    }

    ExecutionRankSelection ExecutionRankSelection::all(int world_size)
    {
        if (world_size <= 0)
            throw std::invalid_argument("Execution selection requires positive discovery size");
        std::vector<int> ranks(static_cast<size_t>(world_size));
        std::iota(ranks.begin(), ranks.end(), 0);
        return ExecutionRankSelection(std::move(ranks));
    }

    std::optional<int> ExecutionRankSelection::executionRank(int discovery_rank) const
    {
        if (discovery_rank < 0) throw std::out_of_range("Negative discovery rank");
        const auto found = std::find(ranks_.begin(), ranks_.end(), discovery_rank);
        if (found == ranks_.end()) return std::nullopt;
        return static_cast<int>(found - ranks_.begin());
    }

    namespace
    {
        /** @return Whitespace-trimmed token; case and spelling remain explicit. */
        std::string_view trimToken(std::string_view value)
        {
            const auto first = value.find_first_not_of(" \t\r\n");
            if (first == std::string_view::npos) return {};
            return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
        }

        /** @brief Parse a hard set without discarding empty or repeated elements. */
        template<class T, class Parser>
        std::vector<T> parseList(std::string_view value, Parser parse)
        {
            std::vector<T> result;
            for (;;)
            {
                const auto comma = value.find(',');
                const auto token = trimToken(value.substr(0, comma));
                if (token.empty()) throw std::invalid_argument("Automatic planning filter contains an empty item");
                const auto item = parse(token);
                if (std::find(result.begin(), result.end(), item) != result.end())
                    throw std::invalid_argument("Automatic planning filter repeats '" + std::string(token) + "'");
                result.push_back(item);
                if (comma == std::string_view::npos) return result;
                value.remove_prefix(comma + 1);
            }
        }

        /** @return Whether a typed backend has a supported production implementation. */
        bool supported(DeviceType backend) noexcept
        {
            return backend == DeviceType::CPU || backend == DeviceType::CUDA || backend == DeviceType::ROCm;
        }

        /** @return Whether a typed strategy names a current execution family. */
        bool supported(OrchestrationStrategy strategy) noexcept
        {
            switch (strategy)
            {
            case OrchestrationStrategy::SingleDevice:
            case OrchestrationStrategy::TensorParallel:
            case OrchestrationStrategy::PipelineParallel:
            case OrchestrationStrategy::ExpertOverlay: return true;
            }
            return false;
        }

        /** @brief Reject invalid typed values, including programmatic callers bypassing parsing. */
        template<class T>
        void validateSet(const std::optional<std::vector<T>> &values)
        {
            if (!values) return;
            if (values->empty()) throw std::invalid_argument("Automatic planning hard constraint cannot be empty");
            for (auto current = values->begin(); current != values->end(); ++current)
            {
                if (!supported(*current)) throw std::invalid_argument("Unsupported automatic planning choice");
                if (std::find(values->begin(), current, *current) != current)
                    throw std::invalid_argument("Automatic planning hard constraint contains duplicates");
            }
        }

        /** @return Membership in an explicit set, or unrestricted when omitted. */
        template<class T>
        bool permitted(const std::optional<std::vector<T>> &values, T value) noexcept
        {
            return supported(value) && (!values || std::find(values->begin(), values->end(), value) != values->end());
        }
    }

    bool AutomaticOrchestrationOptions::specified() const noexcept
    {
        return only_backends || only_strategies || prefer_backend || prefer_strategy || host_participation || workload || device_counts;
    }

    AutomaticOrchestrationRequest::AutomaticOrchestrationRequest(AutomaticOrchestrationOptions options)
        : options_(std::move(options))
    {
        validateSet(options_.only_backends);
        validateSet(options_.only_strategies);
        if (options_.device_counts)
        {
            if (options_.device_counts->empty())
                throw std::invalid_argument("Automatic device counts cannot be empty");
            std::set<DeviceType> seen;
            for (const auto &entry : *options_.device_counts)
                if (entry.count <= 0 || !allows(entry.backend) || !seen.insert(entry.backend).second)
                    throw std::invalid_argument("Automatic device counts require positive counts, distinct permitted backends");
        }
        if (options_.host_participation &&
            *options_.host_participation != AutomaticHostParticipation::BestSubset &&
            *options_.host_participation != AutomaticHostParticipation::AllDiscovered)
            throw std::invalid_argument("Invalid automatic physical-host participation policy");
        if (options_.prefer_backend && !allows(*options_.prefer_backend))
            throw std::invalid_argument("Preferred compute backend is excluded by the hard backend constraint");
        if (options_.prefer_strategy && !allows(*options_.prefer_strategy))
            throw std::invalid_argument("Preferred strategy is excluded by the hard strategy constraint");
    }

    bool AutomaticOrchestrationRequest::allows(DeviceType backend) const noexcept
    {
        return permitted(options_.only_backends, backend);
    }

    bool AutomaticOrchestrationRequest::allows(OrchestrationStrategy strategy) const noexcept
    {
        return permitted(options_.only_strategies, strategy);
    }

    bool AutomaticOrchestrationRequest::allows(
        OrchestrationStrategy strategy, std::span<const DeviceType> participants) const noexcept
    {
        if (!allows(strategy) || participants.empty() ||
            !std::all_of(participants.begin(), participants.end(), [this](DeviceType backend) { return allows(backend); }))
            return false;
        if (options_.device_counts)
            for (const auto &entry : *options_.device_counts)
                if (std::count(participants.begin(), participants.end(), entry.backend) != entry.count)
                    return false;
        return true;
    }

    std::vector<AutomaticBackendDeviceCount> parseAutomaticDeviceCounts(std::string_view value)
    {
        const auto counts = parseList<AutomaticBackendDeviceCount>(value, [](std::string_view token) {
            const auto separator = token.find('=');
            if (separator == std::string_view::npos)
                throw std::invalid_argument("Automatic device counts use backend=count pairs");
            const auto backend = parseOrchestrationComputeBackend(trimToken(token.substr(0, separator)));
            const auto text = trimToken(token.substr(separator + 1));
            if (text.empty())
                throw std::invalid_argument("Automatic device counts require a positive integer");
            int count = 0;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), count);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || count <= 0)
                throw std::invalid_argument("Automatic device counts require a positive integer");
            return AutomaticBackendDeviceCount{backend, count};
        });
        // Use the same admission as programmatic callers; differing counts for
        // one backend are still duplicates, not a range or last-writer-wins.
        AutomaticOrchestrationOptions options;
        options.device_counts = counts;
        (void)AutomaticOrchestrationRequest(std::move(options));
        return counts;
    }

    std::string_view orchestrationStrategyName(OrchestrationStrategy strategy)
    {
        switch (strategy)
        {
        case OrchestrationStrategy::SingleDevice: return "single";
        case OrchestrationStrategy::TensorParallel: return "tp";
        case OrchestrationStrategy::PipelineParallel: return "pp";
        case OrchestrationStrategy::ExpertOverlay: return "expert-overlay";
        }
        throw std::invalid_argument("Invalid orchestration strategy enum");
    }

    OrchestrationStrategy parseOrchestrationStrategy(std::string_view value)
    {
        for (const auto strategy : {OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel,
                                   OrchestrationStrategy::PipelineParallel, OrchestrationStrategy::ExpertOverlay})
            if (value == orchestrationStrategyName(strategy)) return strategy;
        throw std::invalid_argument("Unknown orchestration strategy '" + std::string(value) + "'; use single, tp, pp, expert-overlay");
    }

    AutomaticHostParticipation parseAutomaticHostParticipation(std::string_view value)
    {
        if (value == "best-subset") return AutomaticHostParticipation::BestSubset;
        if (value == "all") return AutomaticHostParticipation::AllDiscovered;
        throw std::invalid_argument("Unknown automatic host policy; use best-subset or all");
    }

    OrchestrationPlanningWorkload parseOrchestrationPlanningWorkload(std::string_view value)
    {
        const auto comma = value.find(',');
        if (comma == std::string_view::npos || value.find(',', comma + 1) != std::string_view::npos)
            throw std::invalid_argument("Planning workload requires prefill,generation token counts");
        const auto positiveInteger = [](std::string_view token) {
            token = trimToken(token);
            if (token.empty()) throw std::invalid_argument("Planning workload contains an empty token count");
            int count = 0;
            const auto result = std::from_chars(token.data(), token.data() + token.size(), count);
            if (result.ec != std::errc{} || result.ptr != token.data() + token.size() || count <= 0)
                throw std::invalid_argument("Planning workload requires positive integer token counts");
            return count;
        };
        return {positiveInteger(value.substr(0, comma)), positiveInteger(value.substr(comma + 1))};
    }

    DeviceType parseOrchestrationComputeBackend(std::string_view value)
    {
        if (value == "cpu") return DeviceType::CPU;
        if (value == "cuda") return DeviceType::CUDA;
        if (value == "rocm") return DeviceType::ROCm;
        throw std::invalid_argument("Unknown compute backend '" + std::string(value) + "'; use cpu, cuda, rocm");
    }

    std::vector<DeviceType> parseOrchestrationBackendList(std::string_view value)
    {
        return parseList<DeviceType>(value, parseOrchestrationComputeBackend);
    }

    std::vector<OrchestrationStrategy> parseOrchestrationStrategyList(std::string_view value)
    {
        return parseList<OrchestrationStrategy>(value, parseOrchestrationStrategy);
    }

    ResolvedOrchestrationIntent resolveOrchestrationIntent(const OrchestrationConfig &config)
    {
        // DeviceSelectionMode already owns the mutually exclusive selector
        // taxonomy. Add policies that constrain layout without naming a device;
        // automatic selection must not silently override those either.
        const bool has_placement = detectDeviceSelectionMode(config) != DeviceSelectionMode::UNSPECIFIED ||
            config.cpu_global_tp_all_local || config.pp_degree > 1 || config.cpu_layers > 0 ||
            config.tp_local_degree > 1 || config.tp_global_degree > 1 || config.heterogeneous_mode ||
            (config.moe_routed_expert_plan && config.moe_routed_expert_plan->enabled);
        const bool applying = config.planning_mode == OrchestrationPlanningMode::Apply ||
            (config.planning_mode == OrchestrationPlanningMode::InferFromPlacement &&
             (has_placement || !config.config_file_path.empty()));

        if (applying)
        {
            if (!has_placement)
                throw std::invalid_argument("Apply requires an explicit device/domain topology; a saved plan cannot silently run auto");
            if (config.automatic_planning.specified())
                throw std::invalid_argument("Automatic planning constraints/preferences cannot accompany an applied topology");
            return ApplyOrchestrationRequest{};
        }
        if (config.planning_mode != OrchestrationPlanningMode::InferFromPlacement &&
            config.planning_mode != OrchestrationPlanningMode::Automatic)
            throw std::invalid_argument("Invalid orchestration planning mode");
        if (has_placement || config.execution_rank_selection)
            throw std::invalid_argument("--auto conflicts with explicit placement; use hard backend/strategy constraints instead");
        if (config.automatic_planning.workload)
            config.automatic_planning.workload->requireFitsContext(config.max_seq_len);
        return AutomaticOrchestrationRequest(config.automatic_planning);
    }
}
