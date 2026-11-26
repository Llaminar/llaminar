/**
 * @file HeterogeneousOverlaySection.cpp
 * @brief Structural validation and declarative sealing for heterogeneous MoE ticket sections.
 */

#include "HeterogeneousOverlaySection.h"
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        const ComputeNode &requireNode(const ComputeGraph &graph,
                                       const GraphNodeRef &handle,
                                       const char *role)
        {
            if (!handle.valid() || !handle.belongsTo(graph))
            {
                throw std::invalid_argument(
                    std::string("HeterogeneousOverlaySection invalid or foreign ") +
                    (role ? role : "node") + " handle");
            }
            const auto *node = graph.getNode(handle.name());
            if (!node || !node->stage)
            {
                throw std::out_of_range(
                    std::string("HeterogeneousOverlaySection missing ") +
                    (role ? role : "node") + " '" + handle.name() + "'");
            }
            return *node;
        }
    } // namespace

    void validateHeterogeneousOverlayTicketSection(
        const ComputeGraph &graph,
        const HeterogeneousOverlayTicketSection &section)
    {
        const auto envelope = graph.nativeCaptureEnvelope();
        if (!requiresHeterogeneousTicketSegmentation(envelope))
        {
            throw std::invalid_argument(
                "HeterogeneousOverlaySection requires an explicit heterogeneous ticket capture envelope");
        }
        if (section.ticket_unit_identity.empty())
        {
            throw std::invalid_argument(
                "HeterogeneousOverlaySection requires a non-empty ticket_unit_identity");
        }

        const auto &router_node = requireNode(graph, section.router, "router");
        const auto &cutpoint_node = requireNode(graph, section.pre_cpu_gpu_cutpoint, "pre_cpu_gpu_cutpoint");
        if (!router_node.device.is_gpu() || cutpoint_node.device != router_node.device)
        {
            throw std::invalid_argument(
                "HeterogeneousOverlaySection router and pre_cpu_gpu_cutpoint must reside on the same GPU");
        }
        if (cutpoint_node.stage->isManualGraphBoundary())
        {
            throw std::invalid_argument(
                "HeterogeneousOverlaySection pre_cpu_gpu_cutpoint '" + cutpoint_node.name +
                "' cannot be a manual boundary");
        }

        const bool is_authority = ownsHeterogeneousTicketBoundary(envelope);
        if (!is_authority && section.manual_host_entry.has_value())
        {
            throw std::logic_error(
                "HeterogeneousOverlaySection forbids manual_host_entry on a follower graph");
        }
        if (!is_authority && !section.rank_batch_dispatch_entries.empty())
        {
            throw std::logic_error(
                "HeterogeneousOverlaySection forbids rank_batch_dispatch_entries on a follower graph");
        }
        if (!is_authority && section.early_trigger.has_value())
        {
            throw std::logic_error(
                "HeterogeneousOverlaySection forbids early_trigger on a follower graph");
        }

        if (section.early_trigger)
        {
            const auto &trigger_node = requireNode(graph, *section.early_trigger, "early_trigger");
            if (trigger_node.device != router_node.device || trigger_node.stage->isManualGraphBoundary())
            {
                throw std::invalid_argument(
                    "HeterogeneousOverlaySection early_trigger must be a captured stage on the router GPU");
            }
            if (trigger_node.stage->isCollectiveStage())
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection early_trigger '" + trigger_node.name +
                    "' cannot be a collective stage");
            }
        }

        for (const auto &branch : section.pre_cpu_overlap_branches)
        {
            const auto &branch_node = requireNode(graph, branch, "pre_cpu_overlap_branch");
            if (branch_node.device != router_node.device || branch_node.stage->isManualGraphBoundary())
            {
                throw std::invalid_argument(
                    "HeterogeneousOverlaySection overlap branch '" + branch_node.name +
                    "' must be a captured GPU stage on the router device");
            }
            if (section.early_trigger)
            {
                if (branch == *section.early_trigger ||
                    graph.hasPath(branch.name(), section.early_trigger->name()))
                {
                    throw std::logic_error(
                        "HeterogeneousOverlaySection forbids GPU overlap branch '" + branch_node.name +
                        "' from preceding early_trigger '" + section.early_trigger->name() +
                        "' (would delay CPU worker wake-up)");
                }
            }
            if (section.post_cpu_join &&
                (branch == *section.post_cpu_join ||
                 graph.hasPath(section.post_cpu_join->name(), branch.name())))
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection overlap branch '" + branch_node.name +
                    "' cannot depend on post_cpu_join '" + section.post_cpu_join->name() + "'");
            }
        }

        if (section.manual_host_entry)
        {
            const auto &host_node = requireNode(graph, *section.manual_host_entry, "manual_host_entry");
            if (!host_node.device.is_cpu() && !host_node.stage->isManualGraphBoundary())
            {
                throw std::invalid_argument(
                    "HeterogeneousOverlaySection manual_host_entry '" + host_node.name +
                    "' must be a CPU or manual graph boundary");
            }
            if (graph.hasPath(host_node.name, cutpoint_node.name))
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection pre_cpu_gpu_cutpoint '" + cutpoint_node.name +
                    "' cannot depend on manual_host_entry '" + host_node.name + "'");
            }
            if (section.early_trigger && graph.hasPath(host_node.name, section.early_trigger->name()))
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection early_trigger cannot depend on manual_host_entry");
            }
        }

        for (const auto &rank_dispatch : section.rank_batch_dispatch_entries)
        {
            const auto &dispatch_node = requireNode(graph, rank_dispatch, "rank_batch_dispatch_entry");
            if (!dispatch_node.device.is_cpu() && !dispatch_node.stage->isManualGraphBoundary())
            {
                throw std::invalid_argument(
                    "HeterogeneousOverlaySection rank_batch_dispatch_entry '" + dispatch_node.name +
                    "' must be a CPU or manual boundary");
            }
            if (graph.hasPath(dispatch_node.name, cutpoint_node.name))
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection pre_cpu_gpu_cutpoint cannot depend on rank_batch_dispatch_entry '" +
                    dispatch_node.name + "'");
            }
        }

        if (section.post_cpu_join)
        {
            const auto &join_node = requireNode(graph, *section.post_cpu_join, "post_cpu_join");
            if (join_node.device != router_node.device || join_node.stage->isManualGraphBoundary())
            {
                throw std::invalid_argument(
                    "HeterogeneousOverlaySection post_cpu_join '" + join_node.name +
                    "' must be a captured stage on the router GPU");
            }
            if (join_node.name == cutpoint_node.name ||
                graph.hasPath(join_node.name, cutpoint_node.name))
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection post_cpu_join '" + join_node.name +
                    "' cannot precede or alias pre_cpu_gpu_cutpoint '" + cutpoint_node.name + "'");
            }
            if (section.manual_host_entry &&
                graph.hasPath(join_node.name, section.manual_host_entry->name()))
            {
                throw std::logic_error(
                    "HeterogeneousOverlaySection manual_host_entry '" + section.manual_host_entry->name() +
                    "' cannot wait on post_cpu_join '" + join_node.name + "'");
            }
        }
    }

    void sealHeterogeneousOverlayTicketSection(
        ComputeGraph &graph,
        const HeterogeneousOverlayTicketSection &section)
    {
        validateHeterogeneousOverlayTicketSection(graph, section);

        // 1. Wire every GPU overlap branch AFTER early_trigger (when present) and INSIDE the pre-CPU unit.
        for (const auto &branch : section.pre_cpu_overlap_branches)
        {
            if (section.early_trigger &&
                !graph.hasPath(section.early_trigger->name(), branch.name()))
            {
                graph.addTypedDependency(
                    branch.name(),
                    section.early_trigger->name(),
                    GraphDependencyKind::ScratchOrder);
            }
            if (branch != section.pre_cpu_gpu_cutpoint)
            {
                graph.addTypedDependency(
                    section.pre_cpu_gpu_cutpoint.name(),
                    branch.name(),
                    GraphDependencyKind::ScratchOrder);
            }
        }

        // 2. Seal the symmetric BeforeManualBoundary ticket-unit cutpoint.
        graph.setHeterogeneousTicketUnitContract(
            section.pre_cpu_gpu_cutpoint.name(),
            GraphHeterogeneousTicketUnitContract{
                .identity = section.ticket_unit_identity,
                .disposition = GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary,
            });

        // 3. Order authority manual host entries after the cutpoint via explicit HostSubmissionOrder
        //    (submission order only; CPU still consumes the early device ticket asynchronously).
        if (section.manual_host_entry)
        {
            graph.addTypedDependency(
                section.manual_host_entry->name(),
                section.pre_cpu_gpu_cutpoint.name(),
                GraphDependencyKind::HostSubmissionOrder);
        }
        for (const auto &rank_dispatch : section.rank_batch_dispatch_entries)
        {
            graph.addTypedDependency(
                rank_dispatch.name(),
                section.pre_cpu_gpu_cutpoint.name(),
                GraphDependencyKind::HostSubmissionOrder);
        }

        // 4. Ensure post-CPU join strictly follows the pre-CPU cutpoint.
        if (section.post_cpu_join &&
            !graph.hasPath(section.pre_cpu_gpu_cutpoint.name(), section.post_cpu_join->name()))
        {
            graph.addTypedDependency(
                section.post_cpu_join->name(),
                section.pre_cpu_gpu_cutpoint.name(),
                GraphDependencyKind::DataFlow);
        }
    }

    void attachPreCPUOverlapBranch(
        ComputeGraph &graph,
        const GraphNodeRef &branch,
        const std::optional<GraphNodeRef> &early_trigger,
        const GraphNodeRef &pre_cpu_cutpoint)
    {
        if (!requiresHeterogeneousTicketSegmentation(graph.nativeCaptureEnvelope()))
        {
            throw std::invalid_argument(
                "attachPreCPUOverlapBranch requires a heterogeneous ticket capture envelope");
        }
        const auto &branch_node = requireNode(graph, branch, "branch");
        const auto &cutpoint_node = requireNode(graph, pre_cpu_cutpoint, "pre_cpu_cutpoint");
        if (!branch_node.device.is_gpu() || branch_node.device != cutpoint_node.device ||
            branch_node.stage->isManualGraphBoundary())
        {
            throw std::invalid_argument(
                "attachPreCPUOverlapBranch requires a captured GPU branch on the cutpoint device");
        }
        if (branch == pre_cpu_cutpoint ||
            !cutpoint_node.heterogeneous_ticket_unit_contract ||
            cutpoint_node.heterogeneous_ticket_unit_contract->disposition !=
                GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary)
        {
            throw std::invalid_argument(
                "attachPreCPUOverlapBranch requires an existing BeforeManualBoundary cutpoint distinct from branch");
        }
        if (early_trigger)
        {
            const auto &trigger_node = requireNode(graph, *early_trigger, "early_trigger");
            if (trigger_node.device != branch_node.device || trigger_node.stage->isManualGraphBoundary())
            {
                throw std::invalid_argument(
                    "attachPreCPUOverlapBranch early_trigger must be a captured GPU stage on the same device");
            }
            if (branch == *early_trigger || graph.hasPath(branch.name(), early_trigger->name()))
            {
                throw std::logic_error(
                    "attachPreCPUOverlapBranch forbids placing GPU branch '" + branch.name() +
                    "' ahead of early_trigger '" + early_trigger->name() + "'");
            }
        }
        if (graph.hasPath(pre_cpu_cutpoint.name(), branch.name()))
        {
            throw std::logic_error(
                "attachPreCPUOverlapBranch forbids placing post-cutpoint node '" + branch.name() +
                "' inside the pre-CPU ticket unit");
        }

        if (early_trigger && !graph.hasPath(early_trigger->name(), branch.name()))
        {
            graph.addTypedDependency(
                branch.name(),
                early_trigger->name(),
                GraphDependencyKind::ScratchOrder);
        }
        graph.addTypedDependency(
            pre_cpu_cutpoint.name(),
            branch.name(),
            GraphDependencyKind::ScratchOrder);
    }

} // namespace llaminar2
