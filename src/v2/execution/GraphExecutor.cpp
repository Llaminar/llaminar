/**
 * @file GraphExecutor.cpp
 * @brief Compute graph execution engine implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "GraphExecutor.h"
#include "StageDumper.h"
#include "AsyncStageDumper.h"
#include "StageCoherence.h"
#include "CollectiveContext.h"
#include "compute_stages/stages/AllreduceStage.h"
#include "compute_stages/stages/AllGatherStage.h"
#include "config/TPDomain.h"
#include "../tensors/TensorVerification.h"
#include "../tensors/GPUTensorVerification.h"
#include "../tensors/TensorClasses.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../utils/KernelProfiler.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
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
    // GraphExecutorStats Implementation
    // =============================================================================

    void GraphExecutorStats::printProfilingSummary(size_t decode_tokens) const
    {
        // Calculate totals
        double total_overhead = overhead.total();
        double total_all = total_execute_ms + total_overhead;

        // Calculate per-token averages if we have decode tokens
        double stages_per_token = decode_tokens > 0 ? static_cast<double>(total_stages_executed) / decode_tokens : 0;
        double execute_per_token = decode_tokens > 0 ? total_execute_ms / decode_tokens : 0;
        double overhead_per_token = decode_tokens > 0 ? total_overhead / decode_tokens : 0;

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    EXECUTOR OVERHEAD PROFILING SUMMARY                       ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

        // Summary line
        std::cout << "║ Total stages executed: " << std::setw(8) << total_stages_executed;
        if (decode_tokens > 0)
        {
            std::cout << "  (" << std::fixed << std::setprecision(1) << stages_per_token << " stages/token)";
        }
        std::cout << std::string(std::max(0, 31 - (decode_tokens > 0 ? 22 : 0)), ' ') << "║\n";

        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  CATEGORY                    │    TOTAL (ms)  │  PER-TOKEN (ms) │    %      ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

        // Helper lambda for printing rows
        auto printRow = [&](const char *name, double value_ms)
        {
            double per_token = decode_tokens > 0 ? value_ms / decode_tokens : 0;
            double pct = total_all > 0 ? (value_ms / total_all) * 100.0 : 0;
            std::cout << "║  " << std::left << std::setw(26) << name << " │ "
                      << std::right << std::setw(13) << std::fixed << std::setprecision(2) << value_ms << " │ "
                      << std::setw(15) << std::fixed << std::setprecision(3) << per_token << " │ "
                      << std::setw(7) << std::fixed << std::setprecision(1) << pct << "%  ║\n";
        };

        // Kernel execution time (the actual work)
        printRow("Kernel Execution", total_execute_ms);

        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  COHERENCE OVERHEAD:                                                         ║\n";
        printRow("  Input Coherence", overhead.input_cohere_ms);
        printRow("  Weight Coherence", overhead.weight_cohere_ms);
        printRow("  Output Allocation", overhead.output_alloc_ms);
        printRow("  Mark Dirty (events)", overhead.mark_dirty_ms);

        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  FRAMEWORK OVERHEAD:                                                         ║\n";
        printRow("  getDumpInfo() calls", overhead.get_dump_info_ms);
        printRow("  extractBuffers() calls", overhead.extract_buffers_ms);
        printRow("  Buffer Verification", overhead.verify_ms);
        printRow("  Snapshot Callbacks", overhead.callback_ms);

        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  STAGE DUMP (if enabled):                                                    ║\n";
        printRow("  Dump Inputs", overhead.dump_input_ms);
        printRow("  Dump Outputs", overhead.dump_output_ms);

        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

        // Totals
        printRow("TOTAL OVERHEAD", total_overhead);
        printRow("TOTAL (kernel + overhead)", total_all);

        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

        // Efficiency metric
        double efficiency = total_all > 0 ? (total_execute_ms / total_all) * 100.0 : 0;
        std::cout << "║  Kernel Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%";
        std::cout << "  (higher = less overhead)                            ║\n";

        if (decode_tokens > 0)
        {
            std::cout << "║  Overhead per token: " << std::fixed << std::setprecision(3) << overhead_per_token << " ms";
            std::cout << "                                             ║\n";
        }

        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << std::endl;
    }

    // =============================================================================
    // ComputeGraph Implementation
    // =============================================================================

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
            return *this;
        }

        auto node = std::make_unique<ComputeNode>(name, std::move(stage), device);
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

        int stage_idx = 0;
        auto prev_time = total_start;

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[GraphExecutor] Invalid node: " << name);
                return false;
            }

            auto stage_start = std::chrono::high_resolution_clock::now();

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[GraphExecutor] Stage failed: " << name);
                return false;
            }

            auto stage_end = std::chrono::high_resolution_clock::now();
            double stage_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();

            // Per-stage timing - TRACE level (use LLAMINAR_EXECUTOR_PROFILING=1 for detailed stats)
            LOG_TRACE("[GraphExecutor] Stage " << stage_idx << "/" << order.size() << ": " << name << " took " << stage_ms << "ms");
            stage_idx++;

            graph.markCompleted(name);
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Wait for any pending async dumps to complete
        if (AsyncStageDumper::isInitialized())
        {
            size_t pending = AsyncStageDumper::pendingTasks();
            if (pending > 0)
            {
                LOG_INFO("[GraphExecutor] Waiting for " << pending << " pending async dumps...");
                AsyncStageDumper::waitForCompletion();
            }
        }

        LOG_DEBUG("[GraphExecutor] Total execution: " << total_ms << "ms for " << order.size() << " stages (" << (total_ms / order.size()) << "ms/stage avg)");

        stats_.total_time_ms += total_ms;
        // Only increment here if profiling is disabled - executeNode already increments when profiling is enabled
        if (!config_.enable_profiling)
        {
            stats_.total_stages_executed += order.size();
        }
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
        const std::unordered_map<DeviceId, IDeviceContext *> &contexts)
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
            if (node->device.is_gpu())
            {
                auto it = contexts.find(node->device);
                if (it != contexts.end())
                {
                    ctx = it->second;
                }
            }

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[GraphExecutor] Stage failed: " << name << " on device " << node->device.to_string());
                return false;
            }

            graph.markCompleted(name);
        }

        return true;
    }

    // =========================================================================
    // Stage Output Debug Printing
    // =========================================================================

    /**
     * @brief Print first N elements of stage outputs for debugging
     *
     * Called AFTER markOutputsDirty() so GPU→host sync has occurred.
     * Controlled by LLAMINAR_STAGE_OUTPUT_PRINT environment variable.
     */
    static void printStageOutputs(const std::string &stage_name, const StageDumpInfo &dump_info)
    {
        const auto &config = debugEnv().stage_output_print;
        if (!config.shouldPrint(stage_name))
        {
            return;
        }

        const int num_elements = config.num_elements;
        const int num_rows = config.num_rows;

        for (const auto &output : dump_info.outputs)
        {
            if (!output.tensor || !output.data)
            {
                continue;
            }

            // Get FP32 data - use TensorBase::data() which handles GPU→host sync
            const float *data = nullptr;

            // Always use TensorBase::data() for coherence-aware access
            auto *tensor_base = dynamic_cast<TensorBase *>(output.tensor);
            if (tensor_base)
            {
                data = tensor_base->data();
            }

            if (!data || output.rows == 0 || output.cols == 0)
            {
                continue;
            }

            const size_t cols = output.cols;
            const size_t rows = output.rows;
            const size_t print_cols = std::min(static_cast<size_t>(num_elements), cols);

            // Build header
            std::ostringstream header;
            header << "[StageOutput] " << stage_name << "/" << (output.name ? output.name : "output")
                   << " [" << rows << "x" << cols << "]";

            // Build first row data
            std::ostringstream first_row;
            first_row << " row[0]: ";
            for (size_t c = 0; c < print_cols; ++c)
            {
                if (c > 0)
                    first_row << ",";
                first_row << data[c];
            }
            if (cols > print_cols)
                first_row << "...";

            // Build last row data if requested
            std::ostringstream last_row;
            if (num_rows > 1 && rows > 1)
            {
                size_t last_idx = rows - 1;
                size_t offset = last_idx * cols;
                last_row << " | row[" << last_idx << "]: ";
                for (size_t c = 0; c < print_cols; ++c)
                {
                    if (c > 0)
                        last_row << ",";
                    last_row << data[offset + c];
                }
                if (cols > print_cols)
                    last_row << "...";
            }

            // Use stream directly with LOG_INFO
            LOG_INFO(header.str() << first_row.str() << last_row.str());
        }
    }

    bool GraphExecutor::executeNode(ComputeNode &node, IDeviceContext *ctx)
    {
        if (!node.stage)
        {
            LOG_ERROR("[GraphExecutor] Node '" << node.name << "' has no stage");
            return false;
        }

        // =========================================================================
        // Transfer Profiling: Set stage context for per-stage transfer tracking
        // Uses RAII to automatically clear context when function exits
        // =========================================================================
        TransferProfiler::StageScope transfer_scope(node.name);

        // =========================================================================
        // Collective Stage Intercept: Use CollectiveContext for GPU-native collectives
        // This bypasses the stage's internal MPI fallback path when CollectiveContext
        // is available, enabling RCCL/NCCL/PCIeBAR backends.
        // =========================================================================
        if (collective_ctx_)
        {
            auto stage_type = node.stage->type();
            if (stage_type == ComputeStageType::ALLREDUCE)
            {
                LOG_DEBUG("[GraphExecutor] Intercepting ALLREDUCE stage '" << node.name << "' via CollectiveContext");
                return executeCollectiveAllreduce(node, ctx);
            }
            else if (stage_type == ComputeStageType::ALLGATHER)
            {
                // Try GPU-native strided allgather (NCCL + CUDA deinterleave kernel)
                // Falls back to stage's MPI path if not CUDA or NCCL unavailable
                LOG_DEBUG("[GraphExecutor] Attempting strided ALLGATHER intercept for '" << node.name << "'");
                if (executeCollectiveStridedAllgather(node, ctx))
                {
                    return true;
                }
                // Fall through to normal execution if strided path not available
                LOG_DEBUG("[GraphExecutor] Strided ALLGATHER not available, using stage execution");
            }
        }

        // Extract layer index from config
        const int layer_idx = config_.current_layer_idx;
        const bool profiling = config_.enable_profiling;

        // Timing variables for phase breakdown (only initialized if profiling enabled)
        std::chrono::high_resolution_clock::time_point phase_start{}, phase_end{};
        double input_cohere_ms = 0.0;
        double weight_cohere_ms = 0.0;
        double output_alloc_ms = 0.0;
        double dump_input_ms = 0.0;
        double execute_ms = 0.0;
        double mark_dirty_ms = 0.0;
        double get_dump_info_ms = 0.0;
        double extract_buffers_ms = 0.0;

        // =========================================================================
        // OPTIMIZATION: Cache getDumpInfo() once at start (avoid 3-4 calls per stage)
        // getDumpInfo() now caches internally, so this is just a reference lookup after first call
        // =========================================================================
        if (profiling)
            phase_start = std::chrono::high_resolution_clock::now();
        const StageDumpInfo &cached_dump_info = node.stage->getDumpInfo();
        if (profiling)
        {
            phase_end = std::chrono::high_resolution_clock::now();
            get_dump_info_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
        }

        // =========================================================================
        // Stage Coherence: Ensure inputs are on target device BEFORE execution
        // =========================================================================
        {
            auto policy = node.stage->coherencePolicy();
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();

            LOG_DEBUG("[GraphExecutor] Stage '" << node.name << "' coherencePolicy=" << toString(policy)
                                                << " target_device=" << target_device.to_string());

            if (policy == CoherencePolicy::INPUT || policy == CoherencePolicy::FULL)
            {
                // Use cached dump_info (no separate getDumpInfo call needed)

                // Cohere inputs (includes extract time)
                if (profiling)
                    phase_start = std::chrono::high_resolution_clock::now();
                auto inputs = extractInputBuffers(cached_dump_info);
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    extract_buffers_ms += std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    phase_start = phase_end;
                }

                if (!cohereInputs(inputs, target_device))
                {
                    LOG_ERROR("[GraphExecutor] Failed to cohere inputs for stage '" << node.name << "'");
                    return false;
                }
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    input_cohere_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }

                // Cohere weights (needed for GPU execution - biases, etc.)
                if (profiling)
                    phase_start = std::chrono::high_resolution_clock::now();
                auto weights = extractWeightBuffers(cached_dump_info);
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    extract_buffers_ms += std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    phase_start = phase_end;
                }

                if (!cohereInputs(weights, target_device)) // Weights are read-only like inputs
                {
                    LOG_ERROR("[GraphExecutor] Failed to cohere weights for stage '" << node.name << "'");
                    return false;
                }
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    weight_cohere_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }
            }

            // For GPU targets, outputs also need GPU buffers allocated before kernel runs
            if (policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL)
            {
                // Use cached dump_info (no separate getDumpInfo call needed)

                if (profiling)
                    phase_start = std::chrono::high_resolution_clock::now();
                auto outputs = extractOutputBuffers(cached_dump_info);
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    extract_buffers_ms += std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    phase_start = phase_end;
                }

                if (!cohereOutputs(outputs, target_device))
                {
                    LOG_ERROR("[GraphExecutor] Failed to allocate output buffers for stage '" << node.name << "'");
                    return false;
                }
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    output_alloc_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }
            }
        }

#if LLAMINAR_ASSERTIONS_ACTIVE
        // ENTRY verification - validate inputs BEFORE execute()
        if (debugEnv().validation.validate_inputs)
        {
            verifyStageEntry(node, layer_idx); // Throws VerificationFailure on error
        }
#endif

        // Check if stage dumping is enabled for this stage
        StageDumpContext dump_ctx;
        const auto &dump_cfg = debugEnv().stage_dump;
        const bool should_dump = StageDumper::shouldDump(
            node.stage.get(),
            node.name, // Pass node name for LLAMINAR_STAGE_DUMP_NAMES filtering
            config_.current_layer_idx,
            config_.current_iteration,
            config_.mpi_rank);

        if (should_dump)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            dump_ctx = StageDumper::beginDump(
                node.stage.get(),
                node.name, // Pass node name for directory naming
                config_.current_layer_idx,
                config_.current_iteration,
                config_.mpi_rank);

            // Use async dumping if enabled (default: true)
            if (dump_cfg.async_dump)
            {
                // Lazy initialization of async dumper
                if (!AsyncStageDumper::isInitialized())
                {
                    AsyncStageDumper::initialize(dump_cfg.async_threads);
                }
                // Enqueue inputs for async writing (fast memcpy only)
                // Use cached dump_info instead of calling getDumpInfo() again
                AsyncStageDumper::enqueueInputs(dump_ctx, cached_dump_info);
            }
            else
            {
                // Synchronous dump (legacy path)
                StageDumper::dumpInputs(dump_ctx, node.stage.get());
            }
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                dump_input_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        if (profiling)
            phase_start = std::chrono::high_resolution_clock::now();
        bool success = node.stage->execute(ctx);
        if (profiling)
        {
            phase_end = std::chrono::high_resolution_clock::now();
            execute_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
        }

        // =========================================================================
        // Stage Coherence: Mark outputs as device-dirty IMMEDIATELY after execution
        // This must happen BEFORE any output data access (dump, verification, callback)
        // so that data() calls will sync from GPU when needed.
        // =========================================================================
        if (success)
        {
            auto policy = node.stage->coherencePolicy();

            if (policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL)
            {
                // Use cached dump_info (no separate getDumpInfo call needed)

                if (profiling)
                    phase_start = std::chrono::high_resolution_clock::now();
                auto outputs = extractOutputBuffers(cached_dump_info);
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    extract_buffers_ms += std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    phase_start = phase_end;
                }

                markOutputsDirty(outputs);
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    mark_dirty_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }

                // Stage output printing (after coherence, so GPU→host sync has occurred)
                printStageOutputs(node.name, cached_dump_info);
            }
        }

        // Timing for dump and validation phases (after core execution)
        double dump_output_ms = 0.0;
        double verify_ms = 0.0;
        double callback_ms = 0.0;

        // Dump outputs after execution (if dumping enabled)
        if (should_dump && success)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();

            if (dump_cfg.async_dump)
            {
                // Enqueue outputs for async writing (fast memcpy only)
                // Use cached dump_info instead of calling getDumpInfo() again
                AsyncStageDumper::enqueueOutputs(dump_ctx, cached_dump_info);
                // Note: finalizeDump not needed for async mode since metadata
                // is written synchronously in beginDump
            }
            else
            {
                // Synchronous dump (legacy path)
                StageDumper::dumpOutputs(dump_ctx, node.stage.get());
                StageDumper::finalizeDump(dump_ctx, execute_ms);
            }
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                dump_output_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        // EXIT verification - validate outputs AFTER execute()
        // (only compiles in Debug/Integration builds with assertions active)
#if LLAMINAR_ASSERTIONS_ACTIVE
        if (success && debugEnv().validation.validate_buffers)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            // New exception-based validation (throws VerificationFailure)
            verifyStageExit(node, layer_idx);

            // Legacy bool-based validation (for compatibility)
            success = validateStageOutputs(node);
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                verify_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }
#endif

        // Invoke snapshot callback if configured (uses cached dump info for efficiency)
        LOG_DEBUG("[GraphExecutor::executeNode] success=" << success << " callback=" << (config_.snapshot_callback ? "set" : "null") << " node=" << node.name);
        if (success && config_.snapshot_callback)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            // IMPORTANT: Sync outputs from GPU before callback reads them
            cached_dump_info.ensureOutputsOnHost();
            LOG_DEBUG("[GraphExecutor::executeNode] Invoking callback for " << node.name);
            config_.snapshot_callback(node.name, cached_dump_info);
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                callback_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        // Log phase breakdown at TRACE level (only for stages taking >1ms total or any phase >0.5ms)
        double total_overhead_ms = input_cohere_ms + weight_cohere_ms + output_alloc_ms + dump_input_ms + mark_dirty_ms + dump_output_ms + verify_ms + callback_ms + get_dump_info_ms + extract_buffers_ms;
        double total_ms = total_overhead_ms + execute_ms;
        if (profiling && (total_ms > 1.0 || input_cohere_ms > 0.5 || weight_cohere_ms > 0.5 ||
                          output_alloc_ms > 0.5 || execute_ms > 0.5 || verify_ms > 0.5 || callback_ms > 0.5))
        {
            LOG_TRACE("[GraphExecutor::PHASES] " << node.name
                                                 << " input_cohere=" << input_cohere_ms << "ms"
                                                 << " weight_cohere=" << weight_cohere_ms << "ms"
                                                 << " output_alloc=" << output_alloc_ms << "ms"
                                                 << " dump_input=" << dump_input_ms << "ms"
                                                 << " execute=" << execute_ms << "ms"
                                                 << " mark_dirty=" << mark_dirty_ms << "ms"
                                                 << " dump_out=" << dump_output_ms << "ms"
                                                 << " verify=" << verify_ms << "ms"
                                                 << " callback=" << callback_ms << "ms"
                                                 << " get_dump_info=" << get_dump_info_ms << "ms"
                                                 << " extract_buffers=" << extract_buffers_ms << "ms"
                                                 << " total=" << total_ms << "ms");
        }

        if (config_.enable_profiling)
        {
            stats_.stage_times_ms[node.name] = total_ms;
            stats_.total_execute_ms += execute_ms;
            stats_.total_stages_executed++;

            // Accumulate overhead breakdown
            stats_.overhead.input_cohere_ms += input_cohere_ms;
            stats_.overhead.weight_cohere_ms += weight_cohere_ms;
            stats_.overhead.output_alloc_ms += output_alloc_ms;
            stats_.overhead.mark_dirty_ms += mark_dirty_ms;
            stats_.overhead.dump_input_ms += dump_input_ms;
            stats_.overhead.dump_output_ms += dump_output_ms;
            stats_.overhead.verify_ms += verify_ms;
            stats_.overhead.callback_ms += callback_ms;
            stats_.overhead.get_dump_info_ms += get_dump_info_ms;
            stats_.overhead.extract_buffers_ms += extract_buffers_ms;

            LOG_DEBUG("[GraphExecutor] Stage '" << node.name << "' took " << total_ms << " ms (execute=" << execute_ms << "ms, overhead=" << total_overhead_ms << "ms)");
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
        // IMPORTANT: Sync outputs from GPU BEFORE reading data
        dump_info.ensureOutputsOnHost();

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
        // NOTE: We intentionally access getDumpInfo() here even though it may trigger
        // GPU→host sync, because we only call this in Debug/Integration builds anyway.
        // The StageDumpInfo provides tensor pointers that we can use for GPU validation.
        auto dump_info = node.stage->getDumpInfo();

        bool all_valid = true;

        // Validate output buffers
        for (const auto &output : dump_info.outputs)
        {
            if (output.rows == 0 || output.cols == 0)
                continue;

            size_t numel = output.rows * output.cols;

            // Check if we have a tensor pointer with GPU data
            // If so, use GPU-side validation to avoid expensive D2H sync
            if (output.tensor)
            {
                auto *base_tensor = dynamic_cast<TensorBase *>(output.tensor);
                if (base_tensor && base_tensor->isDeviceValid())
                {
                    // Skip GPU validation for BAR-backed tensors - they're shared between
                    // CUDA and ROCm devices. Using current_device() returns CUDA device,
                    // but validation might be called from ROCm device thread. Fall through
                    // to host validation which works reliably for mapped/BAR-backed tensors.
                    if (base_tensor->isBARBacked())
                    {
                        LOG_DEBUG("[GraphExecutor] Skipping GPU validation for BAR-backed tensor '"
                                  << output.name << "' - using host validation instead");
                        // Fall through to host validation
                    }
                    else
                    {
                    // Get GPU validator for this device type
                    auto device_opt = base_tensor->current_device();
                    if (device_opt.has_value())
                    {
                        ITensorValidator *validator = getTensorValidator(device_opt->type);
                        if (validator)
                        {
                            const void *device_ptr = base_tensor->gpu_data_ptr();
                            int device_id = device_opt->ordinal;

                            // Launch GPU validation kernel (async)
                            bool launched = false;
                            if (std::string(output.dtype) == "FP32")
                            {
                                launched = validator->validateFP32Async(device_ptr, numel, device_id);
                            }
                            else if (std::string(output.dtype) == "BF16")
                            {
                                launched = validator->validateBF16Async(device_ptr, numel, device_id);
                            }
                            else if (std::string(output.dtype) == "FP16")
                            {
                                launched = validator->validateFP16Async(device_ptr, numel, device_id);
                            }

                            if (launched)
                            {
                                TensorValidationResult result;
                                if (validator->getResult(result))
                                {
                                    if (result.appears_zero && numel > 10)
                                    {
                                        LOG_WARN("[GraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                                           << "' appears to be all zeros (GPU validation)");
                                        if (validation.fail_on_zero)
                                        {
                                            LOG_ERROR("[GraphExecutor] Buffer validation failed: zero tensor detected");
                                            all_valid = false;
                                        }
                                    }

                                    if (result.has_nan || result.has_inf)
                                    {
                                        LOG_WARN("[GraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                                           << "' contains " << result.nan_count << " NaN, "
                                                                           << result.inf_count << " Inf values (GPU validation)");
                                        if (validation.fail_on_nan)
                                        {
                                            LOG_ERROR("[GraphExecutor] Buffer validation failed: NaN/Inf detected");
                                            all_valid = false;
                                        }
                                    }

                                    // Successfully validated on GPU, continue to next output
                                    continue;
                                }
                            }
                            // Fall through to host validation if GPU validation failed to launch
                        }
                    }
                    } // End of non-BAR-backed validation
                }
            }

            // Fallback: Host-side validation (for CPU tensors or when GPU validation unavailable)
            // Need to ensure output is synced to host before reading
            if (output.tensor)
            {
                if (auto *cpu_tensor = dynamic_cast<TensorBase *>(output.tensor))
                {
                    cpu_tensor->ensureOnHost();
                }
            }
            if (!output.data)
                continue;

            if (std::string(output.dtype) == "FP32")
            {
                const float *fp32_data = static_cast<const float *>(output.data);

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
    // Collective Stage Intercept Implementation
    // =============================================================================

    bool GraphExecutor::executeCollectiveAllreduce(ComputeNode &node, IDeviceContext *ctx)
    {
        (void)ctx; // Device context not needed - CollectiveContext handles device

        auto *stage = dynamic_cast<AllreduceStage *>(node.stage.get());
        if (!stage)
        {
            LOG_ERROR("[GraphExecutor] Failed to cast stage '" << node.name << "' to AllreduceStage");
            return false;
        }

        // Get buffer info from stage's dump info
        auto dump_info = stage->getDumpInfo();
        if (dump_info.inputs.empty() || !dump_info.inputs[0].tensor)
        {
            LOG_ERROR("[GraphExecutor] AllreduceStage '" << node.name << "' has no input buffer");
            return false;
        }

        // The allreduce buffer is both input and output (in-place operation)
        ITensor *buffer = dump_info.inputs[0].tensor;
        size_t count = buffer->numel();

        // Determine device where tensor resides
        DeviceId tensor_device = node.device.is_valid() ? node.device : DeviceId::cpu();

        // Extract domain from stage (may be nullptr for legacy path)
        const TPDomain *domain = stage->getDomain();

        // Log timing if profiling is enabled
        auto start = std::chrono::high_resolution_clock::now();

        // Delegate to CollectiveContext - use domain-aware path if domain is set
        bool success;
        if (domain)
        {
            LOG_DEBUG("[GraphExecutor] Executing allreduce in domain: " << domain->name);
            success = collective_ctx_->executeAllreduceInDomain(
                buffer,
                count,
                tensor_device,
                CollectiveOp::ALLREDUCE_SUM,
                domain);
        }
        else
        {
            LOG_DEBUG("[GraphExecutor] Executing allreduce via legacy (no domain) path");
            success = collective_ctx_->executeAllreduce(
                buffer,
                count,
                tensor_device,
                CollectiveOp::ALLREDUCE_SUM);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (config_.enable_profiling)
        {
            stats_.stage_times_ms[node.name] = ms;
            LOG_DEBUG("[GraphExecutor] ALLREDUCE '" << node.name << "' via CollectiveContext took " << ms << " ms");
        }

        if (!success)
        {
            LOG_ERROR("[GraphExecutor] CollectiveContext::executeAllreduce failed for '" << node.name << "'");
        }

        return success;
    }

    bool GraphExecutor::executeCollectiveAllgather(ComputeNode &node, IDeviceContext *ctx)
    {
        (void)ctx; // Device context not needed - CollectiveContext handles device

        auto *stage = dynamic_cast<AllGatherStage *>(node.stage.get());
        if (!stage)
        {
            LOG_ERROR("[GraphExecutor] Failed to cast stage '" << node.name << "' to AllGatherStage");
            return false;
        }

        // Get buffer info from stage's dump info
        auto dump_info = stage->getDumpInfo();

        // AllGather has separate input and output buffers
        ITensor *local_input = nullptr;
        ITensor *full_output = nullptr;

        for (const auto &input : dump_info.inputs)
        {
            if (input.tensor)
            {
                local_input = input.tensor;
                break;
            }
        }

        for (const auto &output : dump_info.outputs)
        {
            if (output.tensor)
            {
                full_output = output.tensor;
                break;
            }
        }

        if (!local_input || !full_output)
        {
            LOG_ERROR("[GraphExecutor] AllGatherStage '" << node.name << "' missing input or output buffer");
            return false;
        }

        // Determine actual sequence length (rows)
        size_t actual_seq_len = local_input->rows();

        // Determine device where tensors reside
        DeviceId tensor_device = node.device.is_valid() ? node.device : DeviceId::cpu();

        // Extract domain from stage (may be nullptr for legacy path)
        const TPDomain *domain = stage->getDomain();

        // Log timing if profiling is enabled
        auto start = std::chrono::high_resolution_clock::now();

        // Delegate to CollectiveContext - use domain-aware path if domain is set
        bool success;
        if (domain)
        {
            LOG_DEBUG("[GraphExecutor] Executing allgather in domain: " << domain->name);
            success = collective_ctx_->executeAllgatherInDomain(
                local_input,
                full_output,
                actual_seq_len,
                tensor_device,
                domain);
        }
        else
        {
            LOG_DEBUG("[GraphExecutor] Executing allgather via legacy (no domain) path");
            success = collective_ctx_->executeAllgather(
                local_input,
                full_output,
                actual_seq_len,
                tensor_device);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (config_.enable_profiling)
        {
            stats_.stage_times_ms[node.name] = ms;
            LOG_DEBUG("[GraphExecutor] ALLGATHER '" << node.name << "' via CollectiveContext took " << ms << " ms");
        }

        if (!success)
        {
            LOG_ERROR("[GraphExecutor] CollectiveContext::executeAllgather failed for '" << node.name << "'");
        }

        return success;
    }

    bool GraphExecutor::executeCollectiveStridedAllgather(ComputeNode &node, IDeviceContext *ctx)
    {
        (void)ctx; // Device context not needed - CollectiveContext handles device

        auto *stage = dynamic_cast<AllGatherStage *>(node.stage.get());
        if (!stage)
        {
            LOG_ERROR("[GraphExecutor] Failed to cast stage '" << node.name << "' to AllGatherStage");
            return false;
        }

        // Get parameters directly from stage
        const auto &params = stage->getParams();

        ITensor *local_input = params.local_input;
        ITensor *full_output = params.full_output;

        if (!local_input || !full_output)
        {
            LOG_DEBUG("[GraphExecutor] AllGatherStage '" << node.name << "' missing input or output buffer");
            return false;
        }

        // Use actual_seq_len from params, fallback to buffer rows
        size_t actual_seq_len = params.actual_seq_len > 0 ? params.actual_seq_len : local_input->rows();

        // Determine device where tensors reside
        DeviceId tensor_device = node.device.is_valid() ? node.device : DeviceId::cpu();

        // Strided allgather only works on CUDA
        if (tensor_device.type != DeviceType::CUDA)
        {
            LOG_DEBUG("[GraphExecutor] Strided allgather requires CUDA device, falling back");
            return false;
        }

        // Log timing if profiling is enabled
        auto start = std::chrono::high_resolution_clock::now();

        // Try strided allgather via CollectiveContext
        // This uses NCCL + CUDA deinterleave kernel to avoid host transfers
        bool success = collective_ctx_->executeStridedAllgather(
            local_input,
            full_output,
            actual_seq_len,
            tensor_device);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (success)
        {
            if (config_.enable_profiling)
            {
                stats_.stage_times_ms[node.name] = ms;
            }
            LOG_DEBUG("[GraphExecutor] Strided ALLGATHER '" << node.name << "' via NCCL took " << ms << " ms");
        }
        else
        {
            LOG_DEBUG("[GraphExecutor] Strided allgather not available for '" << node.name << "'");
        }

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
