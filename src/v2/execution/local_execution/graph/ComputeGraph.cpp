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
 * Region contraction validates every exported completion edge and the entire
 * proposed DAG before retiring stages, including independent output branches.
 */

#include "ComputeGraph.h"
#include "../../moe/MoEExpertOverlayExecutionPlan.h"
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

    ComputeGraph &ComputeGraph::replaceClosedLocalRegion(const std::vector<std::string> &region,
        const std::string &terminal,std::unique_ptr<IComputeStage> replacement)
    {
        return replaceClosedLocalRegion(region,terminal,std::move(replacement),{});
    }

    ComputeGraph &ComputeGraph::replaceClosedLocalRegion(const std::vector<std::string> &region,
        const std::string &terminal,std::unique_ptr<IComputeStage> replacement,
        std::span<const LocalRegionExport> exports)
    {
        return replaceClosedLocalRegion(region,terminal,std::move(replacement),exports,
            LocalRegionCapturePolicy::Unannotated);
    }

    ComputeGraph &ComputeGraph::replaceClosedLocalRegion(const std::vector<std::string> &region,
        const std::string &terminal,std::unique_ptr<IComputeStage> replacement,
        std::span<const LocalRegionExport> exports,LocalRegionCapturePolicy capture_policy)
    {
        const auto require=[](bool condition,const char *message) {
            if(!condition)throw std::invalid_argument(message);
        };
        const std::unordered_set<std::string> members(region.begin(),region.end());
        const bool ticket_local=capture_policy==LocalRegionCapturePolicy::HeterogeneousTicketLocal;
        const bool wave_local=ticket_local || capture_policy==LocalRegionCapturePolicy::WaveAnnotatedLocal;
        require(capture_policy==LocalRegionCapturePolicy::Unannotated ||
            (ticket_local && requiresHeterogeneousTicketSegmentation(nativeCaptureEnvelope())) ||
            (capture_policy==LocalRegionCapturePolicy::WaveAnnotatedLocal &&
                requiresSingleNativeExecutable(nativeCaptureEnvelope())),
            "Ticket-local or wave-annotated fusion requires its matching graph capture envelope");
        auto *sink=getNode(terminal);
        require(region.size()>1 && members.size()==region.size() && members.contains(terminal) &&
            sink && sink->stage && replacement && replacement->device()==sink->device &&
            !replacement->isCollectiveStage(),
            "Local region fusion requires distinct original nodes, one retained sink and a local replacement");
        require(fast_schedule_.empty() && !native_transaction_capture_contract_ &&
            !participant_source_plan_ && participant_source_variants_.empty(),
            "Local region fusion requires cold unsealed construction");
        require(topology_generation_!=std::numeric_limits<std::uint64_t>::max(),
            "Local region fusion topology generation exhausted");
        const auto touched=[&](const std::string &name){return members.contains(name);};
        const auto changesCompletion=[&](const std::string &name) {
            return touched(name) || std::any_of(exports.begin(),exports.end(),[&](const auto &output) {
                return std::find(output.consumers.begin(),output.consumers.end(),name)!=output.consumers.end();
            });
        };
        if(read_ahead_branch_)
        {
            const auto &branch=*read_ahead_branch_;
            require(!changesCompletion(branch.fork_node) && !changesCompletion(branch.producer_node) &&
                !changesCompletion(branch.consume_node) && !changesCompletion(branch.join_node),
                "Local region fusion cannot absorb a declared read-ahead frontier");
        }
        if(participant_auxiliary_range_)
        {
            const auto &range=*participant_auxiliary_range_;
            require(!changesCompletion(range.probe_ready) && !changesCompletion(range.ready_publication) &&
                !changesCompletion(range.main_read) && !changesCompletion(range.retirement) &&
                std::none_of(range.nodes.begin(),range.nodes.end(),changesCompletion),
                "Local region fusion cannot absorb a retained auxiliary frontier");
        }
        for(const auto &name:region)
        {
            const auto *node=getNode(name);
            require(node && node->stage && node->device==sink->device &&
                node->stage->device()==node->device && !node->stage->isCollectiveStage() &&
                !node->stage->isManualGraphBoundary(),
                "Local region fusion cannot absorb missing stages, foreign participants or collectives");
            const auto &wave=node->graph_capture_wave;
            const auto &unit=node->heterogeneous_ticket_unit_contract;
            const bool retained_cutpoint=ticket_local && node==sink && unit &&
                unit->disposition==GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary;
            if(node->completed || node->weights_cohered || node->is_final_output ||
                (wave && (!wave_local || wave->participation!=GraphCaptureWaveParticipation::Active ||
                    !wave->passive_following_identities.empty())) ||
                (unit && !retained_cutpoint) || name==terminal_node_)
                throw std::invalid_argument("Local region fusion cannot absorb live or sealed capture node '"+name+"'");
        }
        std::unordered_map<std::string,std::unordered_set<std::string>> published;
        for(const auto &output:exports)
        {
            require(touched(output.producer) && output.producer!=terminal &&
                !output.consumers.empty() && !published.contains(output.producer),
                "Local region fusion has an invalid or duplicate exported producer");
            auto &readers=published[output.producer];
            for(const auto &name:output.consumers)
            {
                const auto *reader=getNode(name);
                require(!touched(name) && reader && reader->stage &&
                    reader->device==sink->device && reader->stage->device()==sink->device &&
                    !reader->completed && !reader->weights_cohered &&
                    (wave_local || !reader->graph_capture_wave) &&
                    (ticket_local || !reader->heterogeneous_ticket_unit_contract) &&
                    readers.insert(name).second &&
                    std::find(reader->dependencies.begin(),reader->dependencies.end(),output.producer)!=reader->dependencies.end(),
                    "Local region fusion export requires an exact distinct outside reader on this participant");
            }
        }
        // Public tensors remain owned by the replacement. Only explicitly
        // declared readers may acquire its later complete-region publication.
        for(const auto &node:nodes_)
            if(!touched(node->name))
                for(const auto &dependency:node->dependencies)
                    require(!touched(dependency) || dependency==terminal ||
                        (published.contains(dependency) && published.at(dependency).contains(node->name)),
                        "Local region fusion has an external consumer of an interior output");
        std::unordered_set<std::string> ancestors;
        std::vector<std::string> pending{terminal};
        for(const auto &output:exports)pending.push_back(output.producer);
        while(!pending.empty())
        {
            auto name=std::move(pending.back());pending.pop_back();
            if(!ancestors.insert(name).second)continue;
            for(const auto &dependency:getNode(name)->dependencies)
                if(touched(dependency))pending.push_back(dependency);
        }
        require(ancestors.size()==members.size(),
            "Local region fusion contains work disconnected from its declared publications");
        // Validate the original DAG before constructing the replacement edge
        // set; folding must never accidentally turn a broken graph into a valid one.
        (void)getExecutionOrder();
        std::vector<std::string> dependencies;
        std::vector<GraphDependencyKind> dependency_kinds;
        std::unordered_map<std::string,std::size_t> index;
        index.reserve(nodes_.size()-region.size()+1);
        std::vector<std::unique_ptr<ComputeNode>> retained;
        retained.reserve(nodes_.size()-region.size()+1);
        std::vector<std::tuple<ComputeNode*,std::vector<std::string>,std::vector<GraphDependencyKind>>> rewritten;
        rewritten.reserve(nodes_.size()-region.size());
        const auto kindOf = [](const ComputeNode &node, std::size_t idx) {
            return idx < node.dependency_kinds.size() ? node.dependency_kinds[idx] : GraphDependencyKind::DataFlow;
        };
        const auto mergeKind = [](GraphDependencyKind existing, GraphDependencyKind incoming) {
            if (existing == incoming) return existing;
            if (existing == GraphDependencyKind::DataFlow || incoming == GraphDependencyKind::DataFlow)
                return GraphDependencyKind::DataFlow;
            if (existing == GraphDependencyKind::CollectiveOrder || incoming == GraphDependencyKind::CollectiveOrder)
                return GraphDependencyKind::CollectiveOrder;
            return existing;
        };
        for(const auto &node:nodes_)
        {
            if(touched(node->name))
                for(std::size_t di=0; di<node->dependencies.size(); ++di)
                {
                    const auto &dependency = node->dependencies[di];
                    const auto kind = kindOf(*node, di);
                    if(!touched(dependency))
                    {
                        const auto it = std::find(dependencies.begin(),dependencies.end(),dependency);
                        if(it==dependencies.end())
                        {
                            dependencies.push_back(dependency);
                            dependency_kinds.push_back(kind);
                        }
                        else
                        {
                            const auto pos = static_cast<std::size_t>(it - dependencies.begin());
                            dependency_kinds[pos] = mergeKind(dependency_kinds[pos], kind);
                        }
                    }
                }
            if(!touched(node->name) || node.get()==sink)
                index.emplace(node->name,index.size());
            if(!touched(node->name))
            {
                std::vector<std::string> edges;
                std::vector<GraphDependencyKind> kinds;
                edges.reserve(node->dependencies.size());
                kinds.reserve(node->dependencies.size());
                for(std::size_t di=0; di<node->dependencies.size(); ++di)
                {
                    const auto &dependency = node->dependencies[di];
                    const auto kind = kindOf(*node, di);
                    const auto &producer=touched(dependency)?terminal:dependency;
                    const auto it = std::find(edges.begin(),edges.end(),producer);
                    if(it==edges.end())
                    {
                        edges.push_back(producer);
                        kinds.push_back(kind);
                    }
                    else
                    {
                        const auto pos = static_cast<std::size_t>(it - edges.begin());
                        kinds[pos] = mergeKind(kinds[pos], kind);
                    }
                }
                rewritten.emplace_back(node.get(),std::move(edges),std::move(kinds));
            }
        }
        // Exact exports alone do not prove safe contraction: an exported
        // reader could feed another member through an arbitrarily long path.
        // Validate the proposed whole DAG before changing any original edge.
        std::vector<std::vector<std::size_t>> consumers(index.size());
        std::vector<std::size_t> predecessors(index.size(),0);
        const auto addEdges=[&](const std::string &name,const std::vector<std::string> &edges) {
            const auto consumer=index.at(name);
            for(const auto &producer:edges)
            {
                require(index.contains(producer),"Local region fusion has a missing external producer");
                consumers[index.at(producer)].push_back(consumer);
                ++predecessors[consumer];
            }
        };
        addEdges(terminal,dependencies);
        for(const auto &[node,edges,kinds]:rewritten)
        {
            (void)kinds;
            addEdges(node->name,edges);
        }
        std::vector<std::size_t> ready;
        ready.reserve(index.size());
        for(std::size_t i=0;i<predecessors.size();++i)if(predecessors[i]==0)ready.push_back(i);
        for(std::size_t i=0;i<ready.size();++i)
            for(const auto consumer:consumers[ready[i]])if(--predecessors[consumer]==0)ready.push_back(consumer);
        if(ready.size()!=index.size())
        {
            // The original DAG was valid. A cycle in the contraction crosses
            // a member output and returns through an outside dependency.
            for(const auto &dependency:dependencies)
            {
                std::vector<std::string> frontier{dependency};
                std::unordered_map<std::string,std::string> reader{{dependency,{}}};
                for(std::size_t i=0;i<frontier.size();++i)
                {
                    const auto name=frontier[i];
                    if(touched(name))
                    {
                        auto path=name;
                        for(auto next=reader.at(name);!next.empty();next=reader.at(next))
                            path+=" -> "+next;
                        throw std::invalid_argument("Local region fusion at '"+terminal+"' closes a cycle through "+path);
                    }
                    for(const auto &ancestor:getNode(name)->dependencies)
                        if(reader.emplace(ancestor,name).second)frontier.push_back(ancestor);
                }
            }
            throw std::logic_error("Local region fusion topology has inconsistent cycle evidence");
        }
        // Every potentially throwing validation/allocation is complete. The
        // sink keeps its identity and outside edges while removed stages retire.
        sink->stage=std::move(replacement);
        if(wave_local)sink->graph_capture_wave.reset();
        sink->dependencies.swap(dependencies);
        sink->dependency_kinds.swap(dependency_kinds);
        for(auto &[node,edges,kinds]:rewritten)
        {
            node->dependencies.swap(edges);
            node->dependency_kinds.swap(kinds);
        }
        for(auto &node:nodes_)
            if(!touched(node->name) || node.get()==sink)retained.push_back(std::move(node));
        nodes_.swap(retained);node_index_.swap(index);
        cached_order_.clear();order_dirty_=true;cached_execution_stages_.clear();execution_stages_dirty_=true;
        fast_schedule_.clear();noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::replaceProducerAndBypassSuccessor(const std::string &producer,
        const std::string &successor,std::unique_ptr<IComputeStage> replacement)
    {
        auto *first=getNode(producer),*last=getNode(successor);
        const auto fail=[](bool condition,const char *message){if(!condition)throw std::invalid_argument(message);};
        fail(first && last && first!=last && first->stage && last->stage && replacement &&
             first->device==last->device && replacement->device()==first->device,
             "Graph fusion requires two original stages and a matching replacement participant");
        fail(last->dependencies==std::vector<std::string>{producer},
             "Graph fusion successor must have exactly the original producer dependency");
        fail(fast_schedule_.empty() && !first->completed && !last->completed &&
             !first->weights_cohered && !last->weights_cohered && !first->is_final_output && !last->is_final_output,
             "Graph fusion is permitted only during cold construction before execution planning");
        fail(!first->graph_capture_wave && !last->graph_capture_wave &&
             !first->heterogeneous_ticket_unit_contract && !last->heterogeneous_ticket_unit_contract &&
             !native_transaction_capture_contract_ && successor!=terminal_node_ &&
             !participant_source_plan_ && participant_source_variants_.empty(),
             "Graph fusion cannot replace sealed capture, source or terminal metadata");
        const auto touched=[&](const std::string &name){return name==producer || name==successor;};
        if(read_ahead_branch_)
        {
            const auto &branch=*read_ahead_branch_;
            fail(!touched(branch.fork_node) && !touched(branch.producer_node) &&
                 !touched(branch.consume_node) && !touched(branch.join_node),
                 "Graph fusion cannot replace a declared read-ahead frontier");
        }
        if(participant_auxiliary_range_)
        {
            const auto &range=*participant_auxiliary_range_;
            fail(!touched(range.probe_ready) && !touched(range.ready_publication) &&
                 !touched(range.main_read) && !touched(range.retirement) &&
                 std::none_of(range.nodes.begin(),range.nodes.end(),touched),
                 "Graph fusion cannot replace a retained auxiliary frontier");
        }
        fail(topology_generation_!=std::numeric_limits<std::uint64_t>::max(),
             "Graph fusion topology generation exhausted");
        // Allocate every replacement descriptor first. Any validation/allocation
        // failure leaves the original graph, stage pointers and caches intact.
        std::vector<std::vector<std::string>> rewired;rewired.reserve(nodes_.size());
        std::vector<std::vector<GraphDependencyKind>> rewired_kinds;rewired_kinds.reserve(nodes_.size());
        std::unordered_map<std::string,std::size_t> index;index.reserve(node_index_.size()-1);
        for(const auto &node:nodes_)
        {
            if(node.get()==last)continue;
            std::vector<std::string> unique;unique.reserve(node->dependencies.size());
            std::vector<GraphDependencyKind> unique_kinds;unique_kinds.reserve(node->dependencies.size());
            for(std::size_t di=0;di<node->dependencies.size();++di)
            {
                std::string edge=node->dependencies[di]==successor?producer:node->dependencies[di];
                const auto kind=di<node->dependency_kinds.size()?node->dependency_kinds[di]:GraphDependencyKind::DataFlow;
                const auto it=std::find(unique.begin(),unique.end(),edge);
                if(it==unique.end())
                {
                    unique.push_back(std::move(edge));
                    unique_kinds.push_back(kind);
                }
                else if(kind==GraphDependencyKind::DataFlow)
                {
                    unique_kinds[static_cast<std::size_t>(it-unique.begin())]=GraphDependencyKind::DataFlow;
                }
            }
            index.emplace(node->name,rewired.size());
            rewired.push_back(std::move(unique));
            rewired_kinds.push_back(std::move(unique_kinds));
        }
        const auto removed=node_index_.at(successor);
        first->stage=std::move(replacement);
        nodes_.erase(nodes_.begin()+static_cast<std::ptrdiff_t>(removed));
        node_index_.swap(index);
        for(std::size_t i=0;i<nodes_.size();++i)
        {
            nodes_[i]->dependencies.swap(rewired[i]);
            nodes_[i]->dependency_kinds.swap(rewired_kinds[i]);
        }
        cached_order_.clear();order_dirty_=true;cached_execution_stages_.clear();execution_stages_dirty_=true;
        fast_schedule_.clear();noteTopologyMutation();
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

        const std::vector<std::string> leaves = getLeafNodes(GraphLeafScope::MainExecution);
        if (leaves.size() != 1u || leaves.front() != replacement_terminal)
        {
            throw std::logic_error(
                "Replacement heterogeneous graph terminal is not the sole main-execution dependency leaf");
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
        nodes_[it->second]->dependency_kinds.push_back(GraphDependencyKind::DataFlow);
        order_dirty_ = true;
        execution_stages_dirty_ = true;
        fast_schedule_.clear();
        noteTopologyMutation();
        return *this;
    }

    ComputeGraph &ComputeGraph::addTypedDependency(
        const std::string &consumer,
        const std::string &producer,
        GraphDependencyKind kind)
    {
        const auto consumer_it = node_index_.find(consumer);
        if (consumer_it == node_index_.end())
        {
            throw std::out_of_range(
                "ComputeGraph::addTypedDependency consumer node '" + consumer + "' not found");
        }
        const auto producer_it = node_index_.find(producer);
        if (producer_it == node_index_.end())
        {
            throw std::out_of_range(
                "ComputeGraph::addTypedDependency producer node '" + producer + "' not found");
        }
        if (consumer == producer)
        {
            throw std::invalid_argument(
                "ComputeGraph::addTypedDependency rejects self-dependency on '" + consumer + "'");
        }

        auto &consumer_node = *nodes_[consumer_it->second];
        const auto &producer_node = *nodes_[producer_it->second];

        if (kind == GraphDependencyKind::HostSubmissionOrder)
        {
            const bool consumer_is_manual =
                consumer_node.device.is_cpu() ||
                (consumer_node.stage && consumer_node.stage->isManualGraphBoundary());
            const bool producer_is_manual =
                producer_node.stage && producer_node.stage->isManualGraphBoundary();
            if (!consumer_is_manual)
            {
                throw std::invalid_argument(
                    "ComputeGraph::addTypedDependency HostSubmissionOrder cannot target captured device node '" +
                    consumer + "' from '" + producer + "'");
            }
            if (producer_is_manual)
            {
                throw std::invalid_argument(
                    "ComputeGraph::addTypedDependency HostSubmissionOrder cannot source from manual node '" +
                    producer + "' to '" + consumer + "'");
            }
        }

        if (consumer_node.dependency_kinds.size() < consumer_node.dependencies.size())
        {
            consumer_node.dependency_kinds.resize(
                consumer_node.dependencies.size(), GraphDependencyKind::DataFlow);
        }

        const auto existing_it = std::find(
            consumer_node.dependencies.begin(), consumer_node.dependencies.end(), producer);
        if (existing_it != consumer_node.dependencies.end())
        {
            const auto pos = static_cast<std::size_t>(existing_it - consumer_node.dependencies.begin());
            const auto existing_kind = consumer_node.dependency_kinds[pos];
            if (existing_kind == kind || existing_kind == GraphDependencyKind::DataFlow)
                return *this;
            if (kind == GraphDependencyKind::DataFlow)
            {
                consumer_node.dependency_kinds[pos] = GraphDependencyKind::DataFlow;
                noteTopologyMutation();
                return *this;
            }
            throw std::invalid_argument(
                "ComputeGraph::addTypedDependency conflicting edge kinds between consumer '" +
                consumer + "' and producer '" + producer + "'");
        }

        if (hasPath(consumer, producer))
        {
            throw std::invalid_argument(
                "ComputeGraph::addTypedDependency edge '" + consumer + "' <- '" + producer +
                "' would introduce a cycle");
        }

        consumer_node.dependencies.push_back(producer);
        consumer_node.dependency_kinds.push_back(kind);
        order_dirty_ = true;
        execution_stages_dirty_ = true;
        fast_schedule_.clear();
        noteTopologyMutation();
        return *this;
    }

    std::vector<GraphDependencyEdge> ComputeGraph::dependencyEdges(
        const std::string &node_name) const
    {
        const auto *node = getNode(node_name);
        if (!node)
        {
            throw std::out_of_range(
                "ComputeGraph::dependencyEdges node '" + node_name + "' not found");
        }
        std::vector<GraphDependencyEdge> result;
        result.reserve(node->dependencies.size());
        for (std::size_t i = 0; i < node->dependencies.size(); ++i)
        {
            const auto kind = i < node->dependency_kinds.size()
                                  ? node->dependency_kinds[i]
                                  : GraphDependencyKind::DataFlow;
            result.push_back({node->dependencies[i], kind});
        }
        return result;
    }

    std::vector<std::string> ComputeGraph::directConsumersOf(
        const std::string &producer) const
    {
        if (!getNode(producer))
        {
            throw std::out_of_range(
                "ComputeGraph::directConsumersOf producer '" + producer + "' not found");
        }
        std::vector<std::string> consumers;
        for (const auto &name : getExecutionOrder())
        {
            const auto *node = getNode(name);
            if (node &&
                std::find(node->dependencies.begin(), node->dependencies.end(), producer) !=
                    node->dependencies.end())
            {
                consumers.push_back(name);
            }
        }
        return consumers;
    }

    bool ComputeGraph::hasDependency(
        const std::string &consumer,
        const std::string &producer) const noexcept
    {
        const auto *node = getNode(consumer);
        if (!node)
            return false;
        return std::find(node->dependencies.begin(), node->dependencies.end(), producer) !=
               node->dependencies.end();
    }

    bool ComputeGraph::hasPath(
        const std::string &from,
        const std::string &to) const
    {
        if (from.empty() || to.empty() || from == to || !getNode(from) || !getNode(to))
            return false;
        // Walk backward along producer dependencies from `to` to `from`.
        std::vector<std::string> frontier{to};
        std::unordered_set<std::string> visited{to};
        for (std::size_t idx = 0; idx < frontier.size(); ++idx)
        {
            const auto *node = getNode(frontier[idx]);
            if (!node)
                continue;
            for (const auto &dep : node->dependencies)
            {
                if (dep == from)
                    return true;
                if (visited.insert(dep).second)
                    frontier.push_back(dep);
            }
        }
        return false;
    }

    std::vector<LocalRegionExport> ComputeGraph::discoverLocalRegionExports(
        std::span<const std::string> region,
        const std::string &terminal) const
    {
        if (region.empty())
            throw std::invalid_argument("discoverLocalRegionExports requires a non-empty region");
        std::unordered_set<std::string> members;
        members.reserve(region.size());
        for (const auto &name : region)
        {
            if (!getNode(name) || !members.insert(name).second)
            {
                throw std::invalid_argument(
                    "discoverLocalRegionExports requires distinct existing nodes; invalid '" +
                    name + "'");
            }
        }
        if (!members.contains(terminal))
        {
            throw std::invalid_argument(
                "discoverLocalRegionExports requires terminal '" + terminal + "' to be a region member");
        }

        std::unordered_map<std::string, std::vector<std::string>> consumer_lists;
        for (const auto &name : getExecutionOrder())
        {
            if (members.contains(name))
                continue;
            const auto *node = getNode(name);
            if (!node)
                continue;
            for (const auto &dep : node->dependencies)
            {
                if (dep != terminal && members.contains(dep))
                {
                    auto &list = consumer_lists[dep];
                    if (std::find(list.begin(), list.end(), name) == list.end())
                        list.push_back(name);
                }
            }
        }

        std::vector<LocalRegionExport> exports;
        for (const auto &member : region)
        {
            if (member == terminal)
                continue;
            const auto it = consumer_lists.find(member);
            if (it != consumer_lists.end() && !it->second.empty())
                exports.push_back({member, std::move(it->second)});
        }
        return exports;
    }

    void ComputeGraph::validateDeclarativeInvariants() const
    {
        for (const auto &node : nodes_)
        {
            if (!node->dependency_kinds.empty() &&
                node->dependency_kinds.size() != node->dependencies.size())
            {
                throw std::logic_error(
                    "ComputeGraph node '" + node->name +
                    "' has misaligned dependency_kinds metadata");
            }
            if (node->heterogeneous_ticket_unit_contract &&
                !requiresHeterogeneousTicketSegmentation(native_capture_envelope_))
            {
                throw std::logic_error(
                    "ComputeGraph node '" + node->name +
                    "' declares a heterogeneous ticket-unit contract outside a heterogeneous envelope");
            }
            for (std::size_t i = 0; i < node->dependencies.size(); ++i)
            {
                const auto &dep = node->dependencies[i];
                if (dep == node->name)
                {
                    throw std::logic_error(
                        "ComputeGraph node '" + node->name + "' has a self-dependency");
                }
                const auto *producer = getNode(dep);
                if (!producer)
                {
                    throw std::logic_error(
                        "ComputeGraph node '" + node->name +
                        "' depends on missing producer '" + dep + "'");
                }
                const auto kind = i < node->dependency_kinds.size()
                                      ? node->dependency_kinds[i]
                                      : GraphDependencyKind::DataFlow;
                if (kind == GraphDependencyKind::HostSubmissionOrder)
                {
                    const bool consumer_is_manual =
                        node->device.is_cpu() ||
                        (node->stage && node->stage->isManualGraphBoundary());
                    const bool producer_is_manual =
                        producer->stage && producer->stage->isManualGraphBoundary();
                    if (!consumer_is_manual || producer_is_manual)
                    {
                        throw std::logic_error(
                            "ComputeGraph node '" + node->name +
                            "' has invalid HostSubmissionOrder edge from '" + dep + "'");
                    }
                }
            }
        }
        const auto &order = getExecutionOrder();
        if (order.size() != nodes_.size())
        {
            throw std::logic_error(
                "ComputeGraph contains a dependency cycle (" + std::to_string(order.size()) +
                " of " + std::to_string(nodes_.size()) + " nodes topologically ordered)");
        }
    }

    bool ComputeGraph::isRetainedParticipantAuxiliaryNode(const std::string& name) const noexcept
    {
        return participant_auxiliary_range_ && std::find(participant_auxiliary_range_->nodes.begin(),
            participant_auxiliary_range_->nodes.end(),name)!=participant_auxiliary_range_->nodes.end();
    }

    void ComputeGraph::requireRetainedParticipantAuxiliaryRange() const
    {
        if(!participant_auxiliary_range_)return;
        const auto& range=*participant_auxiliary_range_;
        const auto& domain=participant_definition_domain_;
        using Lifecycle=RetainedGraphParticipantAuxiliaryLifecycle;
        const bool current=range.lifecycle==Lifecycle::CurrentDemand;
        const bool future=range.lifecycle==Lifecycle::FutureProducer;
        const bool adopted=range.lifecycle==Lifecycle::AdoptedConsumer;
        const bool discarded=range.lifecycle==Lifecycle::DiscardedConsumer;
        const bool local=adopted || discarded;
        const bool source=range.role==RetainedGraphParticipantAuxiliaryRole::Source;
        if((!current && !future && !local) || (!source && range.role!=RetainedGraphParticipantAuxiliaryRole::Reader) ||
           !domain || !range.participant.is_gpu() || !range.identity || range.nodes.empty() ||
           range.probe_ready.empty() || range.retirement.empty() || range.nodes.back()!=range.retirement ||
           read_ahead_branch_ || source!=(range.participant==domain->source) ||
           (local ? bool(participant_source_plan_) || !range.local_consumption || !range.local_arena :
                    source!=bool(participant_source_plan_) || bool(range.local_consumption) || bool(range.local_arena)))
            throw std::invalid_argument("Participant auxiliary body has an incomplete or foreign lifecycle/source declaration");
        const auto main_node=[&](const std::string& name) {
            const auto* node=getNode(name);
            return node && node->stage && node->device==range.participant && !isRetainedParticipantAuxiliaryNode(name);
        };
        if(!main_node(range.probe_ready) ||
           (current && (range.main_read.empty() || range.probe_ready==range.main_read || !main_node(range.main_read))) ||
           (adopted && (range.main_read!=range.probe_ready || !main_node(range.main_read))) ||
           ((future || discarded) && !range.main_read.empty()) ||
           ((current || future) ? range.ready_publication.empty() || !isRetainedParticipantAuxiliaryNode(range.ready_publication)
                                : !range.ready_publication.empty()))
            throw std::invalid_argument("Participant auxiliary body lost its exact lifecycle main/publication frontier");
        if(local && (range.local_consumption->definitionDomain()!=*domain || !range.local_consumption->captureIdentity()))
            throw std::invalid_argument("Local-only consumption changed its original prepared-source/local declaration");
        std::unordered_set<std::string> visited;
        bool readinessPublished=local;
        for(const auto& name:range.nodes) {
            const auto* node=getNode(name);
            if(!node || !node->stage || node->device!=range.participant || !visited.insert(name).second ||
               node->stage->isManualGraphBoundary() || node->graph_capture_wave || node->heterogeneous_ticket_unit_contract ||
               (!node->stage->isGraphCapturable() && !node->stage->supportsGraphCaptureAfterLaunchPreparation()))
                throw std::invalid_argument("Participant auxiliary body contains an absent, repeated or uncapturable stage: "+name);
            for(const auto& dependency:node->dependencies) {
                if(dependency==name || (!visited.contains(dependency) && dependency!=range.probe_ready &&
                    (!current || dependency!=range.main_read)))
                    throw std::invalid_argument("Participant auxiliary body has an undeclared cross-lane dependency: "+name);
                // Only Current has a later model read. Future producer has no
                // read dependency; a local consumer starts after its real read/discard.
                if(current && dependency==range.main_read && !readinessPublished)
                    throw std::invalid_argument("Participant auxiliary readiness cannot wait for the read it releases: "+name);
            }
            if(name==range.ready_publication)readinessPublished=true;
        }
    }

    void requireRankLocalMixedGPUAuxiliaryExecution(const MoEExpertOverlayExecutionPlan& original)
    {
        const auto& rank = original.current_rank;
        if (original.world_size != 1 || original.continuation_root_rank != 0 ||
            rank.world_rank != 0 || original.rank_plans.size() != 1 ||
            original.rank_plans.front() != rank || !rank.ownsContinuationGraph() ||
            !rank.ownsTransactionAuthority() || !rank.hasRole(OverlayRankRole::ContinuationRoot) ||
            original.continuation_domain.empty() ||
            !rank.ownsDomain(original.continuation_domain) ||
            !rank.cpu_fallback_expert_domains.empty() || !rank.worker_fallback_expert_domains.empty() ||
            rank.loads_cpu_fallback_experts || rank.loads_worker_fallback_experts ||
            original.generationExecutionDomain() != DeviceGenerationExecutionDomain::HostedBoundary)
            throw std::invalid_argument("Native auxiliary parent requires its original world-one mixed-GPU continuation authority");
        bool cuda = false, rocm = false, continuation = false;
        std::unordered_set<std::string> names;
        std::unordered_set<DeviceId> devices;
        for (const auto& domain : original.domains)
        {
            if (domain.name.empty() || !names.insert(domain.name).second || !rank.ownsDomain(domain.name) ||
                (domain.scope != ExecutionDomainScope::SINGLE && domain.scope != ExecutionDomainScope::RANK_LOCAL) ||
                domain.participants.empty())
                throw std::invalid_argument("Native auxiliary parent contains a foreign, distributed or empty original domain");
            if (!domain.primary_world_rank_known || domain.primary_world_rank != 0 ||
                !domain.primary_is_local || !domain.primary_owned_by_current_rank || domain.owner_rank != 0 ||
                std::none_of(domain.participants.begin(), domain.participants.end(), [&](const auto& participant) {
                    return participant.address == domain.primary_participant && participant.local_device == domain.primary_device;
                }))
                throw std::invalid_argument("Native auxiliary parent lost the original domain primary ownership projection");
            continuation = continuation || domain.name == original.continuation_domain;
            for (const auto& participant : domain.participants)
            {
                const auto device = participant.local_device;
                if (!device.is_gpu() || !participant.world_rank_known || participant.world_rank != 0 ||
                    !participant.owned_by_current_rank || !participant.locally_addressable ||
                    participant.participant_index < 0 || !rank.hasLocalDevice(device) ||
                    participant.address.device_type != device.type || participant.address.device_ordinal != device.ordinal)
                    throw std::invalid_argument("Native auxiliary parent requires every original participant to be an owned addressable local GPU");
                cuda = cuda || device.is_cuda();
                rocm = rocm || device.is_rocm();
                devices.insert(device);
            }
        }
        if (!continuation || !cuda || !rocm || rank.local_devices.empty() ||
            names.size() != rank.owned_domains.size() ||
            !names.contains(original.base_model_domain) ||
            (!original.shared_expert_domain.empty() && !names.contains(original.shared_expert_domain)) ||
            std::any_of(rank.local_devices.begin(), rank.local_devices.end(),
                [&](DeviceId device) { return !device.is_gpu() || !devices.contains(device); }))
            throw std::invalid_argument("Native auxiliary parent lost its complete original mixed-GPU membership");
    }

    void ComputeGraph::requireRetainedParticipantNativeAuxiliaryParent() const
    {
        requireRetainedParticipantAuxiliaryRange();
        if (nativeCaptureEnvelope() != GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction ||
            !participant_auxiliary_range_ || !participant_auxiliary_range_->original_execution)
            throw std::invalid_argument("Native auxiliary parent requires its actual original execution and device-owned envelope");
        const auto& range = *participant_auxiliary_range_;
        requireRankLocalMixedGPUAuxiliaryExecution(*range.original_execution);
        const auto& rank = range.original_execution->current_rank;
        if (!rank.hasLocalDevice(range.participant) || !participant_definition_domain_ ||
            !rank.hasLocalDevice(participant_definition_domain_->source))
            throw std::invalid_argument("Native auxiliary parent contains a participant outside its original execution");
        // Existing packet-wave annotations remain descriptive under this
        // envelope; Controller never lowers them into replay boundaries. Actual
        // manual/ticket units and foreign stages remain inadmissible.
        for (const auto& name : getExecutionOrder())
        {
            const auto* node = getNode(name);
            if (!node || !node->stage || node->device != range.participant ||
                node->stage->device() != node->device || node->stage->isManualGraphBoundary() ||
                node->heterogeneous_ticket_unit_contract ||
                (!node->stage->isPassiveGraphCaptureNoOp() && !node->stage->isGraphCapturable() &&
                 !node->stage->supportsGraphCaptureAfterLaunchPreparation()))
                throw std::invalid_argument("Native auxiliary parent cannot contain a foreign, manual or segmented stage: " + name);
        }
    }

    void ComputeGraph::setRetainedParticipantAuxiliaryRange(RetainedGraphParticipantAuxiliaryRange range)
    {
        if(participant_auxiliary_range_)throw std::invalid_argument("Participant auxiliary body cannot be rebound");
        participant_auxiliary_range_=std::move(range);
        try { requireRetainedParticipantAuxiliaryRange(); }
        catch(...) { participant_auxiliary_range_.reset();throw; }
        noteTopologyMutation();
    }

    void ComputeGraph::setReadAheadBranch(GraphReadAheadBranch branch)
    {
        if (read_ahead_phase_identity_!=immutableLookupPhaseIdentity(ForwardExecutionPhase::Decode,ImmutableLookupInvocationSchedule::ResidentSideLane) ||
            read_ahead_branch_ || participant_auxiliary_range_ || !branch.resources ||
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
        participant_auxiliary_range_.reset();
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
        if(participant_auxiliary_range_ && other.participant_auxiliary_range_)
            throw std::invalid_argument("ComputeGraph merge cannot combine two independently declared auxiliary bodies");
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
                    if (nodes_[it->second]->dependency_kinds.size() <
                        nodes_[it->second]->dependencies.size())
                    {
                        nodes_[it->second]->dependency_kinds.resize(
                            nodes_[it->second]->dependencies.size(),
                            GraphDependencyKind::DataFlow);
                    }
                    nodes_[it->second]->dependencies.push_back(connect_from);
                    nodes_[it->second]->dependency_kinds.push_back(GraphDependencyKind::DataFlow);
                }
            }
        }

        if(other.participant_auxiliary_range_) {
            setRetainedParticipantAuxiliaryRange(std::move(*other.participant_auxiliary_range_));
            other.participant_auxiliary_range_.reset();
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

    std::vector<std::string> ComputeGraph::getLeafNodes(GraphLeafScope scope) const
    {
        if (scope != GraphLeafScope::All && scope != GraphLeafScope::MainExecution)
            throw std::invalid_argument("Unknown graph leaf scope");
        const auto included = [&](const std::string& name) {
            return scope == GraphLeafScope::All || !isRetainedParticipantAuxiliaryNode(name);
        };
        // Only edges whose endpoints execute in this projection hide a main leaf.
        std::unordered_set<std::string> has_dependents;
        for (const auto &node : nodes_)
        {
            if (!included(node->name)) continue;
            for (const auto &dep : node->dependencies)
            {
                if (included(dep)) has_dependents.insert(dep);
            }
        }

        // Nodes not in has_dependents are leaves
        std::vector<std::string> leaves;
        for (const auto &node : nodes_)
        {
            if (included(node->name) && has_dependents.find(node->name) == has_dependents.end())
            {
                leaves.push_back(node->name);
            }
        }
        return leaves;
    }

} // namespace llaminar2
