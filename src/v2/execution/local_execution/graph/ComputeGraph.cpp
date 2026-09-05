/**
 * @file ComputeGraph.cpp
 * @brief Compute graph data structure implementation
 * @author David Sanftenberg
 * @date December 2025
 *
 * Extracted from DeviceGraphExecutor.cpp.
 */

#include "ComputeGraph.h"
#include "../../../utils/Logger.h"
#include <queue>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // ComputeGraph Implementation
    // =============================================================================

    void ComputeGraph::noteTopologyMutation()
    {
        if (topology_generation_ == std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error(
                "ComputeGraph topology generation exhausted");
        }
        ++topology_generation_;
    }

    ComputeGraph &ComputeGraph::addNode(const std::string &name,
                                        std::unique_ptr<IComputeStage> stage,
                                        DeviceId device)
    {
        if (node_index_.find(name) != node_index_.end())
        {
            LOG_WARN("[ComputeGraph] Node '" << name << "' already exists, replacing");
            size_t idx = node_index_[name];
            nodes_[idx]->stage = std::move(stage);
            nodes_[idx]->device = device;
            nodes_[idx]->completed = false;
            // A replacement stage must opt into the scheduling contract again;
            // retaining metadata from the previous stage could align the wrong
            // native capture transaction across LocalTP participants.
            nodes_[idx]->graph_capture_wave.reset();
            nodes_[idx]->heterogeneous_ticket_unit_contract.reset();
            cached_execution_stages_.clear();
            execution_stages_dirty_ = true;
            fast_schedule_.clear();
            noteTopologyMutation();
            return *this;
        }

        auto node = std::make_unique<ComputeNode>(name, std::move(stage), device);
        node_index_[name] = nodes_.size();
        nodes_.push_back(std::move(node));
        order_dirty_ = true;
        execution_stages_dirty_ = true;
        fast_schedule_.clear();
        noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::setGraphCaptureWaveContract(
        const std::string &node_name,
        GraphCaptureWaveContract contract)
    {
        auto it = node_index_.find(node_name);
        if (it == node_index_.end())
        {
            throw std::out_of_range(
                "Cannot attach capture-wave contract to missing graph node '" +
                node_name + "'");
        }
        if (contract.identity.empty())
        {
            throw std::invalid_argument(
                "Capture-wave contract for node '" + node_name +
                "' requires a non-empty identity");
        }
        const auto *const stage = nodes_[it->second]->stage.get();
        if (contract.participation ==
                GraphCaptureWaveParticipation::Passive &&
            (!stage || !stage->isPassiveGraphCaptureNoOp()))
        {
            throw std::invalid_argument(
                "Capture-wave contract for node '" + node_name +
                "' declares passive participation, but its stage is not an "
                "immutable graph-capture no-op");
        }

        std::unordered_set<std::string> passive_identities;
        for (const auto &identity : contract.passive_following_identities)
        {
            if (identity.empty())
            {
                throw std::invalid_argument(
                    "Capture-wave contract for node '" + node_name +
                    "' contains an empty passive identity");
            }
            if (!passive_identities.insert(identity).second)
            {
                throw std::invalid_argument(
                    "Capture-wave contract for node '" + node_name +
                    "' repeats passive identity '" + identity + "'");
            }
            if (identity == contract.identity)
            {
                throw std::invalid_argument(
                    "Capture-wave contract for node '" + node_name +
                    "' repeats its own wave identity as a passive follower");
            }
        }

        nodes_[it->second]->graph_capture_wave = std::move(contract);
        noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::setHeterogeneousTicketUnitContract(
        const std::string &node_name,
        GraphHeterogeneousTicketUnitContract contract)
    {
        auto it = node_index_.find(node_name);
        if (it == node_index_.end())
        {
            throw std::out_of_range(
                "Cannot attach heterogeneous ticket-unit boundary to missing graph node '" +
                node_name + "'");
        }
        if (contract.identity.empty())
        {
            throw std::invalid_argument(
                "Heterogeneous ticket-unit boundary for node '" +
                node_name + "' requires a non-empty identity");
        }

        nodes_[it->second]->heterogeneous_ticket_unit_contract =
            std::move(contract);
        noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::moveHeterogeneousTicketTransactionTerminal(
        const std::string &current_terminal,
        const std::string &replacement_terminal)
    {
        const auto current = node_index_.find(current_terminal);
        const auto replacement = node_index_.find(replacement_terminal);
        if (current == node_index_.end())
        {
            throw std::out_of_range(
                "Cannot move heterogeneous ticket terminal from missing graph node '" +
                current_terminal + "'");
        }
        if (replacement == node_index_.end())
        {
            throw std::out_of_range(
                "Cannot move heterogeneous ticket terminal to missing graph node '" +
                replacement_terminal + "'");
        }
        if (!requiresHeterogeneousTicketSegmentation(
                native_capture_envelope_))
        {
            throw std::logic_error(
                "A heterogeneous ticket terminal can move only inside a heterogeneous ticket envelope");
        }
        if (terminalNode() != current_terminal)
        {
            throw std::logic_error(
                "Heterogeneous ticket terminal move does not name the graph's current terminal");
        }

        auto &current_contract =
            nodes_[current->second]->heterogeneous_ticket_unit_contract;
        auto &replacement_contract =
            nodes_[replacement->second]->heterogeneous_ticket_unit_contract;
        if (!current_contract ||
            current_contract->disposition !=
                GraphHeterogeneousTicketUnitDisposition::TransactionTerminal)
        {
            throw std::logic_error(
                "Current heterogeneous graph terminal does not own its transaction-terminal contract");
        }
        if (replacement_contract)
        {
            throw std::logic_error(
                "Replacement heterogeneous graph terminal already owns a ticket-unit contract");
        }

        const std::vector<std::string> leaves = getLeafNodes();
        if (leaves.size() != 1u || leaves.front() != replacement_terminal)
        {
            throw std::logic_error(
                "Replacement heterogeneous graph terminal is not the sole dependency leaf");
        }

        replacement_contract = std::move(current_contract);
        current_contract.reset();
        terminal_node_ = replacement_terminal;
        noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::setNativeCaptureEnvelope(
        GraphNativeCaptureEnvelope envelope)
    {
        if (native_capture_envelope_ == envelope)
            return *this;
        if (native_capture_envelope_ != GraphNativeCaptureEnvelope::Ordinary &&
            envelope != GraphNativeCaptureEnvelope::Ordinary)
        {
            throw std::logic_error(
                "ComputeGraph cannot replace one non-ordinary native-capture envelope with another");
        }
        if (envelope == GraphNativeCaptureEnvelope::Ordinary)
        {
            throw std::logic_error(
                "ComputeGraph cannot weaken an installed native-capture envelope");
        }
        native_capture_envelope_ = envelope;
        noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::addDependency(const std::string &node_name,
                                              const std::string &depends_on)
    {
        auto it = node_index_.find(node_name);
        if (it == node_index_.end())
        {
            LOG_ERROR("[ComputeGraph] Node '" << node_name << "' not found");
            return *this;
        }

        if (node_index_.find(depends_on) == node_index_.end())
        {
            LOG_ERROR("[ComputeGraph] Dependency '" << depends_on << "' not found");
            return *this;
        }

        nodes_[it->second]->dependencies.push_back(depends_on);
        order_dirty_ = true;
        execution_stages_dirty_ = true;
        fast_schedule_.clear();
        noteTopologyMutation();
        return *this;
    }

    const std::vector<std::string> &ComputeGraph::getExecutionOrder() const
    {
        if (!order_dirty_)
            return cached_order_;

        // Kahn's algorithm for topological sort
        std::unordered_map<std::string, int> in_degree;
        std::unordered_map<std::string, std::vector<std::string>> adjacency;

        // Initialize
        for (const auto &node : nodes_)
        {
            in_degree[node->name] = 0;
            adjacency[node->name] = {};
        }

        // Build adjacency list and compute in-degrees
        for (const auto &node : nodes_)
        {
            for (const auto &dep : node->dependencies)
            {
                adjacency[dep].push_back(node->name);
                in_degree[node->name]++;
            }
        }

        // Find all nodes with no dependencies
        std::queue<std::string> ready;
        for (const auto &[name, degree] : in_degree)
        {
            if (degree == 0)
            {
                ready.push(name);
            }
        }

        // Process in topological order
        std::vector<std::string> order;
        order.reserve(nodes_.size());

        while (!ready.empty())
        {
            std::string current = ready.front();
            ready.pop();
            order.push_back(current);

            for (const auto &neighbor : adjacency[current])
            {
                in_degree[neighbor]--;
                if (in_degree[neighbor] == 0)
                {
                    ready.push(neighbor);
                }
            }
        }

        if (order.size() != nodes_.size())
        {
            LOG_ERROR("[ComputeGraph] Cycle detected in graph!");
        }

        cached_order_ = std::move(order);
        order_dirty_ = false;
        return cached_order_;
    }

    const std::vector<IComputeStage *> &ComputeGraph::getExecutionStages()
    {
        if (!execution_stages_dirty_)
            return cached_execution_stages_;

        const std::vector<std::string> &order = getExecutionOrder();
        cached_execution_stages_.clear();
        cached_execution_stages_.reserve(order.size());
        for (const std::string &name : order)
        {
            ComputeNode *node = getNode(name);
            cached_execution_stages_.push_back(
                node && node->stage ? node->stage.get() : nullptr);
        }
        execution_stages_dirty_ = false;
        return cached_execution_stages_;
    }

    std::vector<std::string> ComputeGraph::getReadyNodes() const
    {
        std::vector<std::string> ready;

        for (const auto &node : nodes_)
        {
            if (node->completed)
                continue;

            bool all_deps_complete = true;
            for (const auto &dep : node->dependencies)
            {
                auto dep_node = getNode(dep);
                if (dep_node && !dep_node->completed)
                {
                    all_deps_complete = false;
                    break;
                }
            }

            if (all_deps_complete)
            {
                ready.push_back(node->name);
            }
        }

        return ready;
    }

    ComputeNode *ComputeGraph::getNode(const std::string &name)
    {
        auto it = node_index_.find(name);
        if (it == node_index_.end())
            return nullptr;
        return nodes_[it->second].get();
    }

    const ComputeNode *ComputeGraph::getNode(const std::string &name) const
    {
        auto it = node_index_.find(name);
        if (it == node_index_.end())
            return nullptr;
        return nodes_[it->second].get();
    }

    void ComputeGraph::markCompleted(const std::string &name)
    {
        auto *node = getNode(name);
        if (node)
        {
            node->completed = true;
        }
    }

    void ComputeGraph::reset()
    {
        for (auto &node : nodes_)
        {
            node->completed = false;
        }
    }

    bool ComputeGraph::allCompleted() const
    {
        for (const auto &node : nodes_)
        {
            if (!node->completed)
                return false;
        }
        return true;
    }

    size_t ComputeGraph::totalEstimatedFlops() const
    {
        size_t total = 0;
        for (const auto &node : nodes_)
        {
            if (node->stage)
            {
                total += node->stage->estimatedFlops();
            }
        }
        return total;
    }

    std::unordered_set<std::string> ComputeGraph::collectiveNodeNames() const
    {
        std::unordered_set<std::string> collective_nodes;
        collective_nodes.reserve(nodes_.size());

        /*
         * Walk execution order rather than the private storage vector so this
         * query observes the same finalized node population used by capture
         * planning and fast-schedule construction. The result is a set because
         * callers need constant-time membership while classifying stages.
         */
        for (const auto &node_name : getExecutionOrder())
        {
            const ComputeNode *node = getNode(node_name);
            if (node && node->stage && node->stage->isCollectiveStage())
                collective_nodes.insert(node_name);
        }
        return collective_nodes;
    }

    void ComputeGraph::clear()
    {
        nodes_.clear();
        node_index_.clear();
        cached_order_.clear();
        cached_execution_stages_.clear();
        fast_schedule_.clear();
        order_dirty_ = true;
        execution_stages_dirty_ = true;
        native_capture_envelope_ = GraphNativeCaptureEnvelope::Ordinary;
        noteTopologyMutation();
    }

    void ComputeGraph::buildFastSchedule(const std::unordered_set<std::string> *collective_nodes)
    {
        const auto &order = getExecutionOrder();
        fast_schedule_.clear();
        fast_schedule_.reserve(order.size());

        for (const auto &name : order)
        {
            auto *node = getNode(name);
            if (!node || !node->stage)
                continue;

            bool is_coll = false;
            if (collective_nodes && collective_nodes->count(name) > 0)
            {
                is_coll = true;
            }
            else
            {
                is_coll = node->stage->isCollectiveStage();
            }

            fast_schedule_.push_back({node, is_coll});
        }

        // Mark the last node as needing event-based dirty marking
        if (!fast_schedule_.empty())
        {
            fast_schedule_.back().node->is_final_output = true;
        }
    }

    ComputeGraph &ComputeGraph::merge(ComputeGraph &&other, const std::string &connect_from)
    {
        if (other.nodes_.empty())
        {
            return *this;
        }

        if (native_capture_envelope_ != GraphNativeCaptureEnvelope::Ordinary &&
            other.native_capture_envelope_ !=
                GraphNativeCaptureEnvelope::Ordinary &&
            native_capture_envelope_ != other.native_capture_envelope_)
        {
            throw std::logic_error(
                "ComputeGraph merge received incompatible native-capture envelopes");
        }
        if (native_capture_envelope_ == GraphNativeCaptureEnvelope::Ordinary)
        {
            native_capture_envelope_ = other.native_capture_envelope_;
        }

        // Find root nodes in the source graph (nodes with no dependencies)
        std::vector<std::string> source_roots;
        for (const auto &node : other.nodes_)
        {
            if (node->dependencies.empty())
            {
                source_roots.push_back(node->name);
            }
        }

        // Move all nodes from source to this graph
        for (auto &node : other.nodes_)
        {
            // Check for name collision
            if (node_index_.find(node->name) != node_index_.end())
            {
                LOG_WARN("[ComputeGraph::merge] Node name collision: " << node->name << ", skipping");
                continue;
            }

            size_t idx = nodes_.size();
            node_index_[node->name] = idx;
            nodes_.push_back(std::move(node));
        }

        // If connect_from is specified, connect source roots to it
        if (!connect_from.empty() && node_index_.find(connect_from) != node_index_.end())
        {
            for (const auto &root_name : source_roots)
            {
                auto it = node_index_.find(root_name);
                if (it != node_index_.end())
                {
                    nodes_[it->second]->dependencies.push_back(connect_from);
                }
            }
        }

        // Clear the source graph
        other.nodes_.clear();
        other.node_index_.clear();
        other.cached_order_.clear();
        other.cached_execution_stages_.clear();
        other.fast_schedule_.clear();
        other.order_dirty_ = true;
        other.execution_stages_dirty_ = true;
        other.native_capture_envelope_ =
            GraphNativeCaptureEnvelope::Ordinary;

        order_dirty_ = true;
        execution_stages_dirty_ = true;
        fast_schedule_.clear();
        noteTopologyMutation();
        other.noteTopologyMutation();
        return *this;
    }

    std::vector<std::string> ComputeGraph::getRootNodes() const
    {
        std::vector<std::string> roots;
        for (const auto &node : nodes_)
        {
            if (node->dependencies.empty())
            {
                roots.push_back(node->name);
            }
        }
        return roots;
    }

    std::vector<std::string> ComputeGraph::getLeafNodes() const
    {
        // Build set of all nodes that are depended upon
        std::unordered_set<std::string> has_dependents;
        for (const auto &node : nodes_)
        {
            for (const auto &dep : node->dependencies)
            {
                has_dependents.insert(dep);
            }
        }

        // Nodes not in has_dependents are leaves
        std::vector<std::string> leaves;
        for (const auto &node : nodes_)
        {
            if (has_dependents.find(node->name) == has_dependents.end())
            {
                leaves.push_back(node->name);
            }
        }
        return leaves;
    }

} // namespace llaminar2
