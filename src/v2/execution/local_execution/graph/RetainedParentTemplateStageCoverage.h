/**
 * @file RetainedParentTemplateStageCoverage.h
 * @brief Metadata validation for the main capture units of one retained parent.
 *
 * Main units omit a separately declared finite auxiliary body. Its original
 * declaration must be valid before this helper projects those nodes out of
 * main coverage. The full stored stage pointer and launch-variant roster still
 * covers every declarative node, including that body. Acceptance grants no
 * native executable, attachment, source authority, or permission to launch.
 */
#pragma once

#include "DeviceGraphExecutor.h"
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
/**
 * @brief Validate exact main coverage and the full stored stage identity.
 * @param graph Original complete declarative graph, including auxiliary nodes.
 * @param segments Actual retained main capture units in their original order.
 * @param plan Original full-stage replay identity frozen by the executor.
 * @param error Optional first rejected metadata contract.
 * @return True only for exact main coverage and unchanged full-stage identity.
 * @note This device-free boundary does not authenticate native captures or an
 *       attached body. The caller must separately require the original sealed
 *       attachment and the exact instantiated parent before exporting it.
 */
[[nodiscard]] inline bool validateRetainedParentTemplateStageCoverage(
    const ComputeGraph& graph,
    std::span<const DeviceGraphExecutor::GraphSegment> segments,
    const DeviceGraphExecutor::GraphSegmentCache::RetainedComposedParentReplayPlan& plan,
    std::string* error = nullptr)
{
    const auto reject = [&](const std::string& reason)
    {
        if (error)
            *error = reason;
        return false;
    };
    if (error)
        error->clear();
    const auto& execution_order = graph.getExecutionOrder();
    if (execution_order.empty() || plan.stages.size() != execution_order.size() ||
        plan.stage_variant_signatures.size() != execution_order.size())
        return reject("composed parent has incomplete full source stage identity");

    // Only a structurally authenticated original declaration may be omitted
    // from main units. Its stages remain present in the full identity below.
    if (graph.retainedParticipantAuxiliaryRange())
    {
        try
        {
            graph.requireRetainedParticipantAuxiliaryRange();
        }
        catch (const std::exception& ex)
        {
            return reject(std::string("invalid retained participant auxiliary declaration: ") + ex.what());
        }
    }
    std::vector<std::string_view> main_execution_order;
    main_execution_order.reserve(execution_order.size());
    for (const auto& name : execution_order)
        if (!graph.isRetainedParticipantAuxiliaryNode(name))
            main_execution_order.emplace_back(name);

    size_t covered_stages = 0;
    for (const auto& segment : segments)
    {
        for (const auto& name : segment.stage_names)
            if (covered_stages >= main_execution_order.size() ||
                name != main_execution_order[covered_stages++])
                return reject("composed parent does not cover its complete source graph");
    }
    if (covered_stages != main_execution_order.size())
        return reject("composed parent does not cover its complete source graph");
    for (size_t index = 0; index < execution_order.size(); ++index)
    {
        const auto* node = graph.getNode(execution_order[index]);
        if (!node || !node->stage || node->stage.get() != plan.stages[index] ||
            node->stage->graphCaptureVariantSignature() != plan.stage_variant_signatures[index] ||
            node->stage->graphLaunchPreparationPolicy() == GraphLaunchPreparationPolicy::CaptureAndReplay)
            return reject("composed parent changed a captured stage or requires external replay preparation");
    }
    return true;
}
} // namespace llaminar2
