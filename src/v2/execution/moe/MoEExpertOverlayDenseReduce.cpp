/**
 * @file MoEExpertOverlayDenseReduce.cpp
 * @brief Correctness-first dense partial reduction for MoE expert-overlay tiers.
 */

#include "execution/moe/MoEExpertOverlayDenseReduce.h"

#include "backends/DeviceId.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"
#include "utils/Logger.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace llaminar2
{
    namespace
    {
        std::unordered_map<std::string, const ExpertComputeDomain *> domainsByName(
            const MoEExpertParallelPlan &plan)
        {
            std::unordered_map<std::string, const ExpertComputeDomain *> result;
            for (const auto &domain : plan.domains)
                result.emplace(domain.name, &domain);
            return result;
        }

        std::unordered_map<std::string, const ExpertRoutedTier *> tiersByName(
            const MoEExpertParallelPlan &plan)
        {
            std::unordered_map<std::string, const ExpertRoutedTier *> result;
            for (const auto &tier : plan.routed_tiers)
                result.emplace(tier.name, &tier);
            return result;
        }

    } // namespace

    std::vector<std::string> MoEExpertOverlayDenseReduce::validateRequest(
        const MoEExpertOverlayDenseReduceRequest &request)
    {
        std::vector<std::string> errors;
        auto addError = [&](const std::string &message)
        {
            errors.push_back(message);
        };

        if (!request.plan)
        {
            addError("dense MoE overlay reduce request is missing a plan");
            return errors;
        }

        const auto &plan = *request.plan;
        const auto plan_validation = validateMoEExpertParallelPlan(
            plan,
            MoEExpertParallelValidationOptions{.allow_routed_tensor_parallel_experts = true});
        for (const auto &error : plan_validation.errors)
            addError("plan: " + error);

        if (!plan.isTieredOverlay())
            addError("dense MoE overlay reduce requires an enabled tiered overlay plan");
        if (request.output == nullptr)
            addError("dense MoE overlay reduce request is missing an output tensor");

        const auto domains = domainsByName(plan);
        if (!plan.continuation_domain.empty() && domains.find(plan.continuation_domain) == domains.end())
        {
            addError("continuation domain '" + plan.continuation_domain + "' is not declared in the plan");
        }

        if (request.partials.size() != plan.routed_tiers.size())
        {
            addError("dense MoE overlay reduce requires one partial per routed tier, got " +
                     std::to_string(request.partials.size()) + " partials for " +
                     std::to_string(plan.routed_tiers.size()) + " tiers");
        }

        const auto tiers = tiersByName(plan);
        std::unordered_set<std::string> seen_partial_tiers;
        for (const auto &partial : request.partials)
        {
            if (partial.tensor == nullptr)
            {
                addError("partial for tier '" + partial.tier_name + "' has a null tensor");
            }
            if (partial.tier_name.empty())
            {
                addError("dense MoE overlay partial is missing a tier name");
                continue;
            }
            if (!seen_partial_tiers.insert(partial.tier_name).second)
            {
                addError("duplicate dense MoE overlay partial for tier '" + partial.tier_name + "'");
            }

            const auto tier_it = tiers.find(partial.tier_name);
            if (tier_it == tiers.end())
            {
                addError("partial references unknown routed tier '" + partial.tier_name + "'");
                continue;
            }

            const auto &tier = *tier_it->second;
            if (partial.source_domain.empty())
            {
                addError("partial for tier '" + partial.tier_name + "' is missing a source domain");
            }
            else
            {
                if (domains.find(partial.source_domain) == domains.end())
                {
                    addError("partial for tier '" + partial.tier_name + "' references unknown source domain '" +
                             partial.source_domain + "'");
                }
                if (partial.source_domain != tier.domain)
                {
                    addError("partial for tier '" + partial.tier_name + "' came from domain '" +
                             partial.source_domain + "' but the plan routes that tier through domain '" +
                             tier.domain + "'");
                }
            }
        }

        for (const auto &tier : plan.routed_tiers)
        {
            if (seen_partial_tiers.find(tier.name) == seen_partial_tiers.end())
                addError("missing dense MoE overlay partial for routed tier '" + tier.name + "'");
        }

        return errors;
    }

    bool MoEExpertOverlayDenseReduce::reduceToContinuation(
        const MoEExpertOverlayDenseReduceRequest &request,
        IDeviceContext *ctx)
    {
        const auto errors = validateRequest(request);
        if (!errors.empty())
        {
            for (const auto &error : errors)
                LOG_ERROR("[MoEExpertOverlayDenseReduce] " << error);
            return false;
        }

        MoEExpertParallelReduceStage::Params params;
        params.device_id = DeviceId::cpu();
        params.output = request.output;
        params.rows = request.rows;
        params.cols = request.cols;
        params.mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
        params.continuation_domain = request.plan->continuation_domain;
        params.continuation_device = DeviceId::cpu();
        params.partials.reserve(request.partials.size());
        params.partial_infos.reserve(request.partials.size());
        for (const auto &partial : request.partials)
        {
            params.partials.push_back(partial.tensor);
            params.partial_infos.push_back(MoEExpertParallelReducePartialInfo{
                .name = partial.tier_name,
                .source_domain = partial.source_domain,
                .source_device = DeviceId::cpu(),
            });
        }

        MoEExpertParallelReduceStage stage(std::move(params));
        return stage.execute(ctx);
    }

} // namespace llaminar2