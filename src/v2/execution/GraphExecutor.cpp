/**
 * @file GraphExecutor.cpp
 * @brief Compute graph execution engine implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "GraphExecutor.h"
#include "StageDumper.h"
#include "../tensors/TensorValidation.h"
#include "../tensors/TensorVerification.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include <algorithm>
#include <chrono>
#include <queue>
#include <stdexcept>
#include <unordered_set>

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

    ComputeGraph &ComputeGraph::merge(ComputeGraph &&other, const std::string &connect_from)
    {
        if (other.nodes_.empty())
        {
            return *this;
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

        // Extract layer index from config
        const int layer_idx = config_.current_layer_idx;

#if LLAMINAR_ASSERTIONS_ACTIVE
        // ENTRY verification - validate inputs BEFORE execute()
        if (debugEnv().validation.validate_inputs)
        {
            verifyStageEntry(node, layer_idx); // Throws VerificationFailure on error
        }
#endif

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

        // EXIT verification - validate outputs AFTER execute()
        // (only compiles in Debug/Integration builds with assertions active)
#if LLAMINAR_ASSERTIONS_ACTIVE
        if (success && debugEnv().validation.validate_buffers)
        {
            // New exception-based validation (throws VerificationFailure)
            verifyStageExit(node, layer_idx);

            // Legacy bool-based validation (for compatibility)
            success = validateStageOutputs(node);
        }
#endif

        // Invoke snapshot callback if configured (uses same dump info for efficiency)
        if (success && config_.snapshot_callback)
        {
            auto dump_info = node.stage->getDumpInfo();
            config_.snapshot_callback(node.name, dump_info);
        }

        return success;
    }

    // =============================================================================
    // Buffer Validation (Debug/Integration Builds Only)
    // =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE

    void GraphExecutor::verifyStageEntry(const ComputeNode &node, int layer_idx)
    {
        using namespace verification;

        const auto &validation = debugEnv().validation;
        auto dump_info = node.stage->getDumpInfo();

        // Build verification config from global settings
        VerificationConfig vconfig;
        vconfig.sample_rows = validation.sample_rows;
        vconfig.check_null = true;
        vconfig.check_nan = validation.fail_on_nan;
        vconfig.check_inf = validation.fail_on_nan; // Inf is also bad
        vconfig.check_all_zero = false;             // Zero inputs may be valid (first layer residual)
        vconfig.dump_on_failure = validation.dump_on_failure;

        // Verify all inputs (NaN/Inf/null checks)
        for (const auto &input : dump_info.inputs)
        {
            if (!input.data)
                continue; // Null inputs checked separately if needed

            auto result = verifyRawBuffer(
                input.data, input.rows, input.cols,
                input.name, input.dtype, vconfig);

            if (!result.passed)
            {
                LOG_ERROR("[VERIFY] ENTRY FAILED: layer=" << layer_idx
                                                          << " stage=" << node.name
                                                          << " tensor=" << result.tensor_name
                                                          << " reason=" << result.error_reason);

                // Dump all buffers for debugging
                std::string dump_path;
                if (vconfig.dump_on_failure)
                {
                    dump_path = dumpStageBuffers(node.name, layer_idx, "ENTRY", dump_info,
                                                 result.tensor_name, result.error_reason);
                    LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
                }

                // Throw exception with full context
                throw VerificationFailure(node.name, layer_idx, "ENTRY",
                                          result.tensor_name, result.error_reason, dump_path);
            }
        }

        // =====================================================================
        // Phase 3: Automatic Layout Validation (declarative)
        // If stage provides LayoutExpectation, validate all buffers with layouts
        // =====================================================================
        if (validation.validate_inputs)
        {
            auto layout_expect = node.stage->getLayoutExpectation();
            if (layout_expect.is_set())
            {
                auto buf_reqs = node.stage->getBufferRequirements();
                for (const auto &buf : buf_reqs.buffers)
                {
                    // Only validate buffers with declared layouts
                    if (buf.expected_layout == TensorLayout::UNKNOWN)
                        continue;

                    // Only validate INPUT buffers at entry
                    if (buf.role != BufferRole::INPUT && buf.role != BufferRole::INOUT)
                        continue;

                    auto result = validateBufferLayoutByShape(
                        buf.shape, buf.name.c_str(),
                        buf.expected_layout, layout_expect);

                    if (!result.passed)
                    {
                        LOG_ERROR("[VERIFY] LAYOUT FAILED: layer=" << layer_idx
                                                                   << " stage=" << node.name
                                                                   << " buffer=" << buf.name
                                                                   << " reason=" << result.error_reason);

                        std::string dump_path;
                        if (vconfig.dump_on_failure)
                        {
                            dump_path = dumpStageBuffers(node.name, layer_idx, "ENTRY_LAYOUT", dump_info,
                                                         buf.name, result.error_reason);
                        }

                        throw VerificationFailure(node.name, layer_idx, "ENTRY_LAYOUT",
                                                  buf.name, result.error_reason, dump_path);
                    }
                }
            }
        }
    }

    void GraphExecutor::verifyStageExit(const ComputeNode &node, int layer_idx)
    {
        using namespace verification;

        const auto &validation = debugEnv().validation;
        auto dump_info = node.stage->getDumpInfo();

        // Build verification config from global settings
        VerificationConfig vconfig;
        vconfig.sample_rows = validation.sample_rows;
        vconfig.check_null = true;
        vconfig.check_nan = validation.fail_on_nan;
        vconfig.check_inf = validation.fail_on_nan;
        vconfig.dump_on_failure = validation.dump_on_failure;

        // All-zero output check: enabled by config UNLESS stage explicitly allows zero outputs
        // Most stages should never produce all-zero outputs (indicates bugs).
        // Stages like KVCacheGatherStage can override allowsZeroOutput() to return true.
        vconfig.check_all_zero = validation.fail_on_zero && !node.stage->allowsZeroOutput();

        // Verify all outputs (NaN/Inf/null checks)
        for (const auto &output : dump_info.outputs)
        {
            if (!output.data)
                continue;

            auto result = verifyRawBuffer(
                output.data, output.rows, output.cols,
                output.name, output.dtype, vconfig);

            if (!result.passed)
            {
                LOG_ERROR("[VERIFY] EXIT FAILED: layer=" << layer_idx
                                                         << " stage=" << node.name
                                                         << " tensor=" << result.tensor_name
                                                         << " reason=" << result.error_reason);

                // Dump all buffers for debugging
                std::string dump_path;
                if (vconfig.dump_on_failure)
                {
                    dump_path = dumpStageBuffers(node.name, layer_idx, "EXIT", dump_info,
                                                 result.tensor_name, result.error_reason);
                    LOG_ERROR("[VERIFY] Buffers dumped to: " << dump_path);
                }

                // Throw exception with full context
                throw VerificationFailure(node.name, layer_idx, "EXIT",
                                          result.tensor_name, result.error_reason, dump_path);
            }
        }

        // =====================================================================
        // Phase 3: Automatic Layout Validation (declarative)
        // Validate OUTPUT buffers with declared layouts at stage exit
        // =====================================================================
        if (validation.validate_buffers)
        {
            auto layout_expect = node.stage->getLayoutExpectation();
            if (layout_expect.is_set())
            {
                auto buf_reqs = node.stage->getBufferRequirements();
                for (const auto &buf : buf_reqs.buffers)
                {
                    // Only validate buffers with declared layouts
                    if (buf.expected_layout == TensorLayout::UNKNOWN)
                        continue;

                    // Only validate OUTPUT buffers at exit
                    if (buf.role != BufferRole::OUTPUT && buf.role != BufferRole::INOUT)
                        continue;

                    auto result = validateBufferLayoutByShape(
                        buf.shape, buf.name.c_str(),
                        buf.expected_layout, layout_expect);

                    if (!result.passed)
                    {
                        LOG_ERROR("[VERIFY] LAYOUT FAILED: layer=" << layer_idx
                                                                   << " stage=" << node.name
                                                                   << " buffer=" << buf.name
                                                                   << " reason=" << result.error_reason);

                        std::string dump_path;
                        if (vconfig.dump_on_failure)
                        {
                            dump_path = dumpStageBuffers(node.name, layer_idx, "EXIT_LAYOUT", dump_info,
                                                         buf.name, result.error_reason);
                        }

                        throw VerificationFailure(node.name, layer_idx, "EXIT_LAYOUT",
                                                  buf.name, result.error_reason, dump_path);
                    }
                }
            }
        }
    }

    bool GraphExecutor::validateStageOutputs(const ComputeNode &node)
    {
        if (!node.stage)
            return true;

        const auto &validation = debugEnv().validation;

        // Get stage's dump info to access output buffers
        auto dump_info = node.stage->getDumpInfo();

        bool all_valid = true;

        // Validate output buffers
        for (const auto &output : dump_info.outputs)
        {
            if (!output.data || output.rows == 0 || output.cols == 0)
                continue;

            // Check for zero tensor (uninitialized buffer)
            // Only check FP32 outputs for now
            if (std::string(output.dtype) == "FP32")
            {
                const float *fp32_data = static_cast<const float *>(output.data);
                size_t numel = output.rows * output.cols;

                // Quick zero check: sample first, middle, last elements
                bool appears_zero = true;
                if (numel > 0 && fp32_data[0] != 0.0f)
                    appears_zero = false;
                if (numel > 1 && fp32_data[numel / 2] != 0.0f)
                    appears_zero = false;
                if (numel > 2 && fp32_data[numel - 1] != 0.0f)
                    appears_zero = false;

                // Full check if samples all zero
                if (appears_zero && numel > 3)
                {
                    size_t sample_stride = std::max(size_t(1), numel / 100);
                    for (size_t i = 0; i < numel; i += sample_stride)
                    {
                        if (fp32_data[i] != 0.0f)
                        {
                            appears_zero = false;
                            break;
                        }
                    }
                }

                if (appears_zero)
                {
                    LOG_WARN("[GraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                       << "' appears to be all zeros (likely uninitialized)");

                    if (validation.fail_on_zero)
                    {
                        LOG_ERROR("[GraphExecutor] Buffer validation failed: zero tensor detected");
                        all_valid = false;
                    }
                }

                // Check for NaN/Inf
                bool has_nan_inf = false;
                for (size_t i = 0; i < numel && !has_nan_inf; i += std::max(size_t(1), numel / 100))
                {
                    if (std::isnan(fp32_data[i]) || std::isinf(fp32_data[i]))
                    {
                        has_nan_inf = true;
                    }
                }

                if (has_nan_inf)
                {
                    LOG_WARN("[GraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                       << "' contains NaN or Inf values");

                    if (validation.fail_on_nan)
                    {
                        LOG_ERROR("[GraphExecutor] Buffer validation failed: NaN/Inf detected");
                        all_valid = false;
                    }
                }
            }
        }

        return all_valid;
    }
#endif // LLAMINAR_ASSERTIONS_ACTIVE

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
