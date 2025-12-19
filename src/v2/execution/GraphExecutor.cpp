/**
 * @file GraphExecutor.cpp
 * @brief Compute graph execution engine implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "GraphExecutor.h"
#include "StageDumper.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include <algorithm>
#include <chrono>
#include <queue>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // ExecutionMode Helpers
    // =============================================================================

    const char *executionModeName(ExecutionMode mode)
    {
        switch (mode)
        {
        case ExecutionMode::SEQUENTIAL:
            return "SEQUENTIAL";
        case ExecutionMode::PARALLEL:
            return "PARALLEL";
        case ExecutionMode::PIPELINED:
            return "PIPELINED";
        default:
            return "UNKNOWN";
        }
    }

    // =============================================================================
    // ComputeGraph Implementation
    // =============================================================================

    ComputeGraph &ComputeGraph::addNode(const std::string &name,
                                        std::unique_ptr<IComputeStage> stage,
                                        int device_idx)
    {
        if (node_index_.find(name) != node_index_.end())
        {
            LOG_WARN("[ComputeGraph] Node '" << name << "' already exists, replacing");
            size_t idx = node_index_[name];
            nodes_[idx]->stage = std::move(stage);
            nodes_[idx]->device_idx = device_idx;
            nodes_[idx]->completed = false;
            return *this;
        }

        auto node = std::make_unique<ComputeNode>(name, std::move(stage), device_idx);
        node_index_[name] = nodes_.size();
        nodes_.push_back(std::move(node));
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
        return *this;
    }

    std::vector<std::string> ComputeGraph::getExecutionOrder() const
    {
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

        return order;
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

    void ComputeGraph::clear()
    {
        nodes_.clear();
        node_index_.clear();
    }

    // =============================================================================
    // GraphExecutor Implementation
    // =============================================================================

    GraphExecutor::GraphExecutor(const GraphExecutorConfig &config)
        : config_(config) {}

    GraphExecutor::~GraphExecutor() = default;

    // =============================================================================
    // Execution
    // =============================================================================

    bool GraphExecutor::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[GraphExecutor] Null device context");
            return false;
        }

        if (graph.size() == 0)
        {
            return true; // Empty graph is success
        }

        graph.reset();

        switch (config_.mode)
        {
        case ExecutionMode::SEQUENTIAL:
            return executeSequential(graph, ctx);
        case ExecutionMode::PARALLEL:
            return executeParallel(graph, ctx);
        case ExecutionMode::PIPELINED:
            LOG_WARN("[GraphExecutor] Pipelined mode not yet implemented, using sequential");
            return executeSequential(graph, ctx);
        default:
            LOG_ERROR("[GraphExecutor] Unknown execution mode");
            return false;
        }
    }

    bool GraphExecutor::executeSequential(ComputeGraph &graph, IDeviceContext *ctx)
    {
        auto order = graph.getExecutionOrder();

        auto total_start = std::chrono::high_resolution_clock::now();

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[GraphExecutor] Invalid node: " << name);
                return false;
            }

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[GraphExecutor] Stage failed: " << name);
                return false;
            }

            graph.markCompleted(name);
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        stats_.total_time_ms += total_ms;
        stats_.total_stages_executed += order.size();
        stats_.total_flops += graph.totalEstimatedFlops();

        return true;
    }

    bool GraphExecutor::executeParallel(ComputeGraph &graph, IDeviceContext *ctx)
    {
        auto total_start = std::chrono::high_resolution_clock::now();

        while (!graph.allCompleted())
        {
            auto ready = graph.getReadyNodes();

            if (ready.empty() && !graph.allCompleted())
            {
                LOG_ERROR("[GraphExecutor] Deadlock detected in graph");
                return false;
            }

            // Execute all ready nodes
            // In a true parallel implementation, this would dispatch to different threads
            // For now, execute sequentially (parallel execution requires more infrastructure)
            for (const auto &name : ready)
            {
                auto *node = graph.getNode(name);
                if (!node || !node->stage)
                    continue;

                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[GraphExecutor] Stage failed: " << name);
                    return false;
                }

                graph.markCompleted(name);
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        stats_.total_time_ms += total_ms;
        stats_.total_flops += graph.totalEstimatedFlops();

        return true;
    }

    bool GraphExecutor::executeMultiDevice(
        ComputeGraph &graph,
        const std::unordered_map<int, IDeviceContext *> &contexts)
    {

        if (contexts.empty())
        {
            LOG_ERROR("[GraphExecutor] No device contexts provided");
            return false;
        }

        // Default context for nodes without explicit device assignment
        IDeviceContext *default_ctx = nullptr;
        for (const auto &[idx, ctx] : contexts)
        {
            default_ctx = ctx;
            break;
        }

        graph.reset();
        auto order = graph.getExecutionOrder();

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;

            // Find appropriate context for this node's device
            IDeviceContext *ctx = default_ctx;
            if (node->device_idx >= 0)
            {
                auto it = contexts.find(node->device_idx);
                if (it != contexts.end())
                {
                    ctx = it->second;
                }
            }

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[GraphExecutor] Stage failed: " << name << " on device " << node->device_idx);
                return false;
            }

            graph.markCompleted(name);
        }

        return true;
    }

    bool GraphExecutor::executeNode(ComputeNode &node, IDeviceContext *ctx)
    {
        if (!node.stage)
        {
            LOG_ERROR("[GraphExecutor] Node '" << node.name << "' has no stage");
            return false;
        }

        // Check if stage dumping is enabled for this stage
        StageDumpContext dump_ctx;
        const bool should_dump = StageDumper::shouldDump(
            node.stage.get(),
            config_.current_layer_idx,
            config_.current_iteration,
            config_.mpi_rank);

        if (should_dump)
        {
            dump_ctx = StageDumper::beginDump(
                node.stage.get(),
                config_.current_layer_idx,
                config_.current_iteration,
                config_.mpi_rank);
            StageDumper::dumpInputs(dump_ctx, node.stage.get());
        }

        auto start = std::chrono::high_resolution_clock::now();

        bool success = node.stage->execute(ctx);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (config_.enable_profiling)
        {
            stats_.stage_times_ms[node.name] = ms;

            LOG_DEBUG("[GraphExecutor] Stage '" << node.name << "' took " << ms << " ms");
        }

        // Dump outputs after execution (if dumping enabled)
        if (should_dump && success)
        {
            StageDumper::dumpOutputs(dump_ctx, node.stage.get());
            StageDumper::finalizeDump(dump_ctx, ms);
        }

        // Invoke snapshot callback if configured (uses same dump info for efficiency)
        if (success && config_.snapshot_callback)
        {
            auto dump_info = node.stage->getDumpInfo();
            config_.snapshot_callback(node.name, dump_info);
        }

        return success;
    }

    // =============================================================================
    // Buffer Management
    // =============================================================================

    bool GraphExecutor::executeWithBufferManagement(ComputeGraph &graph, IDeviceContext *ctx)
    {
        if (!buffer_manager_)
        {
            LOG_ERROR("[GraphExecutor] executeWithBufferManagement called without buffer manager set");
            return false;
        }

        LOG_DEBUG("[GraphExecutor] Allocating buffers for graph...");

        // Allocate all buffers based on stage requirements
        if (!buffer_manager_->allocateForGraph(graph))
        {
            LOG_ERROR("[GraphExecutor] Failed to allocate buffers for graph");
            return false;
        }

        LOG_DEBUG("[GraphExecutor] Allocated " << buffer_manager_->bufferCount()
                                               << " buffers (" << (buffer_manager_->totalAllocatedBytes() / 1024.0 / 1024.0)
                                               << " MB)");

        // Execute the graph with normal execution path
        bool success = execute(graph, ctx);

        // Note: Buffers are intentionally NOT released here
        // Caller can retrieve them via buffer_manager_->getBuffer()
        // Caller is responsible for releasing via buffer_manager_->releaseAll()

        return success;
    }

    // =============================================================================
    // Workspace Management
    // =============================================================================

    float *GraphExecutor::getTemporaryBuffer(size_t elements)
    {
        size_t needed = elements * 2; // Double for gate+up buffers

        if (needed > temp_buffer_size_)
        {
            temp_buffer_.resize(needed);
            temp_buffer_size_ = needed;
        }

        return temp_buffer_.data();
    }

} // namespace llaminar2
