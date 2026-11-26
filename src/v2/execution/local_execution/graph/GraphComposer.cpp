/**
 * @file GraphComposer.cpp
 * @brief Implementation of the strict declarative GraphComposer API.
 */

#include "GraphComposer.h"
#include <stdexcept>
#include <unordered_set>

namespace llaminar2
{
    GraphNodeRef GraphComposer::ref(const std::string &existing_node) const
    {
        if (existing_node.empty() || !graph_.getNode(existing_node))
        {
            throw std::out_of_range(
                "GraphComposer::ref cannot bind missing graph node '" + existing_node + "'");
        }
        return GraphNodeRef(&graph_, existing_node);
    }

    std::optional<GraphNodeRef> GraphComposer::optionalRef(const std::string &existing_node) const
    {
        if (existing_node.empty())
            return std::nullopt;
        return ref(existing_node);
    }

    void GraphComposer::requireMember(const GraphNodeRef &handle, const char *role) const
    {
        const std::string role_name = role ? std::string(role) : std::string("node");
        if (!handle.valid())
        {
            throw std::invalid_argument(
                "GraphComposer " + role_name + " handle is empty or unbound");
        }
        if (!handle.belongsTo(graph_))
        {
            throw std::invalid_argument(
                "GraphComposer " + role_name + " '" + handle.name() +
                "' belongs to a different ComputeGraph");
        }
        if (!graph_.getNode(handle.name()))
        {
            throw std::out_of_range(
                "GraphComposer " + role_name + " '" + handle.name() +
                "' no longer exists in its ComputeGraph");
        }
    }

    GraphNodeRef GraphComposer::add(GraphNodeDecl decl)
    {
        if (decl.name.empty())
            throw std::invalid_argument("GraphComposer::add requires a non-empty node name");
        if (!decl.stage)
            throw std::invalid_argument(
                "GraphComposer::add requires a non-null stage for node '" + decl.name + "'");
        if (graph_.getNode(decl.name) != nullptr)
        {
            throw std::invalid_argument(
                "GraphComposer::add rejects duplicate node name '" + decl.name + "'");
        }

        const bool consumer_is_manual =
            decl.device.is_cpu() || decl.stage->isManualGraphBoundary();

        // Validate every declared producer handle and semantic edge rule BEFORE mutating the graph.
        std::vector<std::pair<std::string, GraphDependencyKind>> edges;
        edges.reserve(decl.data_deps.size() + decl.order_deps.size() +
                      decl.host_submission_deps.size() + decl.collective_deps.size());

        const auto collect = [&](const std::vector<GraphNodeRef> &refs,
                                 GraphDependencyKind kind,
                                 const char *role)
        {
            for (const auto &dep : refs)
            {
                requireMember(dep, role);
                if (dep.name() == decl.name)
                {
                    throw std::invalid_argument(
                        "GraphComposer::add rejects self-dependency on '" + decl.name + "'");
                }
                if (kind == GraphDependencyKind::HostSubmissionOrder)
                {
                    const auto *producer = graph_.getNode(dep.name());
                    const bool producer_is_manual =
                        producer && producer->stage && producer->stage->isManualGraphBoundary();
                    if (!consumer_is_manual)
                    {
                        throw std::invalid_argument(
                            "GraphComposer::add rejects HostSubmissionOrder targeting captured device node '" +
                            decl.name + "' from '" + dep.name() + "'");
                    }
                    if (producer_is_manual)
                    {
                        throw std::invalid_argument(
                            "GraphComposer::add rejects HostSubmissionOrder from manual producer '" +
                            dep.name() + "' to '" + decl.name + "'");
                    }
                }
                edges.emplace_back(dep.name(), kind);
            }
        };

        collect(decl.data_deps, GraphDependencyKind::DataFlow, "data_dep");
        collect(decl.order_deps, GraphDependencyKind::ScratchOrder, "order_dep");
        collect(decl.host_submission_deps, GraphDependencyKind::HostSubmissionOrder, "host_submission_dep");
        collect(decl.collective_deps, GraphDependencyKind::CollectiveOrder, "collective_dep");

        if (decl.ticket_unit)
        {
            if (decl.ticket_unit->identity.empty())
            {
                throw std::invalid_argument(
                    "GraphComposer::add ticket_unit on '" + decl.name + "' requires a non-empty identity");
            }
            if (!requiresHeterogeneousTicketSegmentation(graph_.nativeCaptureEnvelope()))
            {
                throw std::invalid_argument(
                    "GraphComposer::add rejects ticket_unit on '" + decl.name +
                    "' outside a heterogeneous ticket capture envelope");
            }
        }
        if (decl.capture_wave && decl.capture_wave->identity.empty())
        {
            throw std::invalid_argument(
                "GraphComposer::add capture_wave on '" + decl.name + "' requires a non-empty identity");
        }

        // All validation passed; commit atomically.
        graph_.addNode(decl.name, std::move(decl.stage), decl.device);
        for (const auto &[producer_name, kind] : edges)
            graph_.addTypedDependency(decl.name, producer_name, kind);
        if (decl.capture_wave)
            graph_.setGraphCaptureWaveContract(decl.name, std::move(*decl.capture_wave));
        if (decl.ticket_unit)
            graph_.setHeterogeneousTicketUnitContract(decl.name, std::move(*decl.ticket_unit));

        return GraphNodeRef(&graph_, std::move(decl.name));
    }

    void GraphComposer::connect(const GraphNodeRef &consumer,
                                const GraphNodeRef &producer,
                                GraphDependencyKind kind)
    {
        requireMember(consumer, "consumer");
        requireMember(producer, "producer");
        graph_.addTypedDependency(consumer.name(), producer.name(), kind);
    }

    void GraphComposer::requireNoPath(const GraphNodeRef &from,
                                      const GraphNodeRef &to,
                                      const char *diagnostic) const
    {
        requireMember(from, "from");
        requireMember(to, "to");
        if (from == to || graph_.hasPath(from.name(), to.name()))
        {
            throw std::logic_error(
                std::string(diagnostic ? diagnostic : "Forbidden graph dependency path detected") +
                " (forbidden path: '" + from.name() + "' -> ... -> '" + to.name() + "')");
        }
    }

    void GraphComposer::requireIndependent(const GraphNodeRef &a,
                                           const GraphNodeRef &b,
                                           const char *diagnostic) const
    {
        requireNoPath(a, b, diagnostic);
        requireNoPath(b, a, diagnostic);
    }

} // namespace llaminar2
