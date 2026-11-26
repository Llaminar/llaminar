/**
 * @file LocalMegakernelLowering.cpp
 * @brief Implementation of declarative local-region contraction for megakernels.
 */

#include "LocalMegakernelLowering.h"
#include <stdexcept>
#include <unordered_set>

namespace llaminar2
{
    GraphNodeRef lowerLocalMegakernelRegion(ComputeGraph &graph, LocalMegakernelRegionSpec spec)
    {
        GraphComposer composer(graph);
        composer.requireMember(spec.terminal, "megakernel terminal");
        if (!spec.replacement)
        {
            throw std::invalid_argument(
                "lowerLocalMegakernelRegion requires a non-null replacement stage");
        }
        if (spec.members.size() < 2u)
        {
            throw std::invalid_argument(
                "lowerLocalMegakernelRegion requires at least two distinct region members");
        }

        std::vector<std::string> member_names;
        member_names.reserve(spec.members.size());
        std::unordered_set<std::string> seen;
        seen.reserve(spec.members.size());
        for (const auto &member : spec.members)
        {
            composer.requireMember(member, "megakernel member");
            if (!seen.insert(member.name()).second)
            {
                throw std::invalid_argument(
                    "lowerLocalMegakernelRegion contains duplicate member '" + member.name() + "'");
            }
            member_names.push_back(member.name());
        }
        if (!seen.contains(spec.terminal.name()))
        {
            throw std::invalid_argument(
                "lowerLocalMegakernelRegion terminal '" + spec.terminal.name() +
                "' must be included in members");
        }

        const auto exports = graph.discoverLocalRegionExports(member_names, spec.terminal.name());
        const auto policy = spec.capture_policy.value_or(graph.defaultLocalRegionCapturePolicy());
        graph.replaceClosedLocalRegion(
            member_names,
            spec.terminal.name(),
            std::move(spec.replacement),
            exports,
            policy);
        return composer.ref(spec.terminal.name());
    }

} // namespace llaminar2
