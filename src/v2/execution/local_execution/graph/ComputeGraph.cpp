/**
 * @file ComputeGraph.cpp
 * @brief Compute graph data structure implementation
 * @author David Sanftenberg
 * @date December 2025
 *
 * Extracted from DeviceGraphExecutor.cpp. Geometry-only workspace topologies
 * carry a monotone execution purpose through moves/merges, so an allocator's
 * declaration can never become a retained executable by losing its input tag.
 * Merging retains a source fragment's storage lifetime together with its
 * participant metadata; query-only fragments never acquire a source lane.
 */

#include "ComputeGraph.h"
#include "../../../backends/ReadAheadNativeResourceFamily.h"
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

    ComputeGraph &ComputeGraph::setNativeTransactionCaptureContract(GraphNativeTransactionCaptureContract contract)
    {
        if (contract.identity.empty() || contract.terminal_node.empty() ||
            contract.terminal_node != terminal_node_ || !getNode(contract.terminal_node) ||
            requiresHeterogeneousTicketSegmentation(native_capture_envelope_) ||
            (native_transaction_capture_contract_ && *native_transaction_capture_contract_ != contract))
            throw std::invalid_argument("Native transaction capture requires its exact sealed terminal, identity and complete envelope");
        if (native_transaction_capture_contract_) return *this;
        native_transaction_capture_contract_ = std::move(contract);
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
        if (native_transaction_capture_contract_ && requiresHeterogeneousTicketSegmentation(envelope))
            throw std::invalid_argument("Whole native transaction cannot acquire a heterogeneous ticket envelope");
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

    void ComputeGraph::setReadAheadBranch(GraphReadAheadBranch branch)
    {
        if (read_ahead_phase_identity_!=immutableLookupPhaseIdentity(ForwardExecutionPhase::Decode,ImmutableLookupInvocationSchedule::ResidentSideLane) ||
            read_ahead_branch_ || !branch.resources ||
            branch.fork_node == branch.producer_node || branch.fork_node == branch.consume_node ||
            branch.producer_node == branch.consume_node)
            throw std::invalid_argument("Read-ahead requires one immutable admitted branch");
        auto *fork=getNode(branch.fork_node), *producer=getNode(branch.producer_node), *consumer=getNode(branch.consume_node);
        if (!fork || !producer || !consumer || !fork->stage || !producer->stage || !consumer->stage ||
            producer->dependencies.size()!=1 || producer->dependencies.front()!=branch.fork_node ||
            std::find(consumer->dependencies.begin(),consumer->dependencies.end(),branch.producer_node)==consumer->dependencies.end())
            throw std::invalid_argument("Read-ahead requires Hash -> sole gather dependency and explicit Consume dependency");
        for (auto* node : {fork,producer,consumer})
            if (node->device != branch.resources->device() || node->stage->isCollectiveStage() || node->stage->isManualGraphBoundary())
                throw std::invalid_argument("Read-ahead branch must remain on its exact native participant");
        if (!branch.resources->stream() || !branch.resources->forkEvent() || !branch.resources->joinEvent() ||
            !branch.resources->primaryStream() || !branch.resources->captureIdentity())
            throw std::invalid_argument("Read-ahead native resources are incomplete");
        read_ahead_branch_=std::move(branch);
        noteTopologyMutation();
    }

    void ComputeGraph::resolveReadAheadJoin()
    {
        if(!read_ahead_branch_)return;
        auto& branch=*read_ahead_branch_;
        if(!branch.join_node.empty())return; // Cold plan; replay performs no descriptor allocation.
        std::vector<ReadAheadJoinFrontier> frontiers;
        const auto& order=getExecutionOrder();frontiers.reserve(order.size());
        for(const auto& name:order)
        {
            const auto* node=getNode(name);auto kind=ReadAheadJoinFrontier::Kind::Ordinary;
            if(node->heterogeneous_ticket_unit_contract)
            {
                switch(node->heterogeneous_ticket_unit_contract->disposition)
                {
                case GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary:kind=ReadAheadJoinFrontier::Kind::BeforeDeclaredHostBoundary;break;
                case GraphHeterogeneousTicketUnitDisposition::TransactionTerminal:kind=ReadAheadJoinFrontier::Kind::TransactionTerminal;break;
                default:throw std::logic_error("Unknown original native-unit frontier disposition");
                }
            }
            frontiers.push_back({name,kind});
        }
        const auto selected=resolveReadAheadJoinAnchor(frontiers,branch.fork_node,branch.producer_node,branch.consume_node,branch.join_policy);
        if(!branch.join_node.empty() && branch.join_node!=selected)
            throw std::logic_error("Read-ahead native join anchor changed after planning");
        if(branch.join_node.empty()){branch.join_node=selected;noteTopologyMutation();}
    }

    void ComputeGraph::planReadAheadRecordingStream(void* recording)
    {
        if(!read_ahead_branch_)return;
        resolveReadAheadJoin();
        const auto& resources=read_ahead_branch_->resources;
        if(recording==resources->stream())throw std::logic_error("Read-ahead primary recording aliases its side lane");
        if(read_ahead_recording_binding_.plan(recording,resources->primaryStream()))noteTopologyMutation();
    }

    void* ComputeGraph::executionStreamForNode(const std::string& name,void* primary) const
    {
        if (!primary) throw std::invalid_argument("Graph stage binding requires an exact non-null parent stream");
        if (!read_ahead_branch_) return primary;
        const auto& branch=*read_ahead_branch_;
        read_ahead_recording_binding_.require(primary);
        return name==branch.producer_node ? branch.resources->stream() : primary;
    }

    std::uint64_t ComputeGraph::readAheadCaptureIdentity() const noexcept
    {
        if(!read_ahead_branch_)return read_ahead_phase_identity_;
        const auto& branch=*read_ahead_branch_;
        auto identity=branch.resources->captureIdentity()^read_ahead_phase_identity_^read_ahead_recording_binding_.identity();
        for(const auto* name:{&branch.fork_node,&branch.producer_node,&branch.consume_node,&branch.join_node})
        {
            for(unsigned char c:*name){identity^=c;identity*=1099511628211ull;}
            identity^=0xffu;identity*=1099511628211ull;
        }
        identity^=static_cast<std::uint64_t>(branch.join_policy);identity*=1099511628211ull;
        return identity;
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

        if (read_ahead_branch_)
        {
            // Submission priority is distinct from a dependency: gathering must
            // enqueue early without making independent trunk kernels wait.
            const auto &branch=*read_ahead_branch_;
            auto producer=std::find(order.begin(),order.end(),branch.producer_node);
            if (producer==order.end()) throw std::logic_error("Read-ahead producer is absent from executable order");
            order.erase(producer);
            auto fork=std::find(order.begin(),order.end(),branch.fork_node);
            if (fork==order.end()) throw std::logic_error("Read-ahead fork is absent from executable order");
            order.insert(fork+1,branch.producer_node);
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

    bool ComputeGraph::requiresCollectiveRuntimeCapture(const std::unordered_set<std::string>* named)const{
        if(named)for(const auto& name:*named){const auto* node=getNode(name);if(!node || !node->stage)return true;}
        for(const auto& name:getExecutionOrder()){
            const auto* node=getNode(name);
            if(node && node->stage && node->stage->requiresCollectiveRuntimeCapture(named && named->count(name)))return true;
        }
        return false;
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
        participant_source_plan_.reset();
        participant_definition_domain_.reset();
        participant_source_variants_.clear();
        storage_boundary_.reset();
        read_ahead_branch_.reset();
        read_ahead_phase_identity_=0;
        read_ahead_recording_binding_={};
        purpose_=ExecutionPurpose::Executable;
        node_index_.clear();
        cached_order_.clear();
        cached_execution_stages_.clear();
        fast_schedule_.clear();
        order_dirty_ = true;
        execution_stages_dirty_ = true;
        native_transaction_capture_contract_.reset();
        native_capture_envelope_ = GraphNativeCaptureEnvelope::Ordinary;
        executable_memory_class_ = GPUGraphExecutableClass::General;
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
        if(other.read_ahead_branch_)
            throw std::logic_error("Read-ahead graphs must be retained as complete native parents, not merged into another topology");
        // Source fragments own an exclusive storage lane; query-only fragments
        // intentionally do not. Transfer the original optional ownership with
        // the topology instead of making every model builder rebind a lane.
        // Validate all ownership conflicts first: a rejected reader/source merge
        // must not install a lane before discovering a foreign participant domain.
        if(storage_boundary_ && other.storage_boundary_ && storage_boundary_!=other.storage_boundary_)
            throw std::invalid_argument("ComputeGraph merge changed its original immutable storage boundary");
        if(participant_definition_domain_ && other.participant_definition_domain_ &&
            *participant_definition_domain_!=*other.participant_definition_domain_)
            throw std::invalid_argument("ComputeGraph merge changed its original participant definition domain");
        if(participant_source_plan_ && other.participant_source_plan_ && participant_source_plan_!=other.participant_source_plan_)
            throw std::logic_error("One retained source parent cannot hide multiple participant invocation plans");
        if (!participant_source_variants_.empty() || !other.participant_source_variants_.empty())
            throw std::logic_error("A sealed expert source-family association cannot be merged into another graph");
        if(other.storage_boundary_)setRetainedStorageBoundary(other.storage_boundary_);
        if(other.participant_definition_domain_)setRetainedParticipantDefinitionDomain(*other.participant_definition_domain_);
        if(other.participant_source_plan_)
            setRetainedParticipantSourcePlan(other.participant_source_plan_);

        if(other.isWorkspaceDeclaration())markWorkspaceDeclaration();
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

        // A standalone child identity cannot name the larger enclosing graph.
        // The outer builder may declare its own complete transaction after merge.
        native_transaction_capture_contract_.reset();
        other.native_transaction_capture_contract_.reset();

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
