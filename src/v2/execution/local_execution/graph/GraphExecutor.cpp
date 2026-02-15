/**
 * @file GraphExecutor.cpp
 * @brief Compute graph execution engine implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "GraphExecutor.h"
#include "../../debug/StageDumper.h"
#include "../../debug/AsyncStageDumper.h"
#include "../coherence/StageCoherence.h"
#include "../collective/CollectiveContext.h"
#include "../../compute_stages/stages/AllreduceStage.h"
#include "../../compute_stages/stages/AllGatherStage.h"
#include "../../../config/TPDomain.h"
#include "../../../tensors/TensorVerification.h"
#include "../../../tensors/GPUTensorVerification.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#ifdef HAVE_ROCM
#include "../../../backends/rocm/HipDeviceGuard.h"
#endif
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "fort.hpp"
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace llaminar2
{

    // =========================================================================
    // GraphSegmentCache — capture stream management
    // =========================================================================

    bool GraphExecutor::GraphSegmentCache::ensureCaptureStream(IWorkerGPUContext *ctx)
    {
        if (capture_stream)
            return true;
        if (!ctx)
        {
            LOG_ERROR("[GraphSegmentCache] No GPU context for stream creation");
            return false;
        }
        capture_stream = ctx->createStream();
        if (!capture_stream)
        {
            LOG_ERROR("[GraphSegmentCache] Failed to create capture stream");
            return false;
        }
        gpu_ctx_ref = ctx;
        LOG_DEBUG("[GraphSegmentCache] Created local capture stream");
        return true;
    }

    void GraphExecutor::GraphSegmentCache::destroyCaptureStream()
    {
        if (!capture_stream)
            return;
        if (gpu_ctx_ref)
            gpu_ctx_ref->destroyStream(capture_stream);
        capture_stream = nullptr;
        gpu_ctx_ref = nullptr;
    }

    bool GraphExecutor::GraphSegmentCache::ensureSyncEvent(IWorkerGPUContext *ctx)
    {
        if (sync_event)
            return true;
        if (!ctx)
            return false;
        sync_event = ctx->createEvent();
        if (!sync_event)
        {
            LOG_ERROR("[GraphSegmentCache] Failed to create sync event");
            return false;
        }
        return true;
    }

    void GraphExecutor::GraphSegmentCache::destroySyncEvent()
    {
        if (!sync_event)
            return;
        if (gpu_ctx_ref)
            gpu_ctx_ref->destroyEvent(sync_event);
        sync_event = nullptr;
    }

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
        (void)execute_per_token; // Used in efficiency section

        // Helper to format a double with fixed precision
        auto fmt = [](double val, int prec) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        // Helper to format percentage
        auto pct = [&](double val) -> std::string
        {
            double p = total_all > 0 ? (val / total_all) * 100.0 : 0;
            return fmt(p, 1) + "%";
        };

        // Helper to format per-token value
        auto per_tok = [&](double val) -> std::string
        {
            if (decode_tokens == 0)
                return "-";
            return fmt(val / decode_tokens, 3);
        };

        // Build the table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title row (spans all columns)
        table << "EXECUTOR OVERHEAD PROFILING SUMMARY" << "" << "" << "" << fort::endr;
        table[0][0].set_cell_span(4);
        table[0][0].set_cell_text_align(fort::text_align::center);
        table.row(0).set_cell_content_fg_color(fort::color::light_cyan);

        // Summary info
        {
            std::ostringstream oss;
            oss << "Total stages executed: " << total_stages_executed;
            if (decode_tokens > 0)
                oss << "  (" << fmt(stages_per_token, 1) << " stages/token)";
            table << oss.str() << "" << "" << "" << fort::endr;
            table[1][0].set_cell_span(4);
        }

        // Header row
        table << fort::header << "CATEGORY" << "TOTAL (ms)" << "PER-TOKEN (ms)" << "%" << fort::endr;

        // Column alignments
        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);

        // Kernel execution
        table << "Kernel Execution" << fmt(total_execute_ms, 2) << per_tok(total_execute_ms) << pct(total_execute_ms) << fort::endr;

        // Separator before coherence
        table << fort::separator;
        table << "COHERENCE OVERHEAD:" << "" << "" << "" << fort::endr;
        table << "  Input Coherence" << fmt(overhead.input_cohere_ms, 2) << per_tok(overhead.input_cohere_ms) << pct(overhead.input_cohere_ms) << fort::endr;
        table << "  Weight Coherence" << fmt(overhead.weight_cohere_ms, 2) << per_tok(overhead.weight_cohere_ms) << pct(overhead.weight_cohere_ms) << fort::endr;
        table << "  Output Allocation" << fmt(overhead.output_alloc_ms, 2) << per_tok(overhead.output_alloc_ms) << pct(overhead.output_alloc_ms) << fort::endr;
        table << "  Mark Dirty (events)" << fmt(overhead.mark_dirty_ms, 2) << per_tok(overhead.mark_dirty_ms) << pct(overhead.mark_dirty_ms) << fort::endr;

        // Framework overhead
        table << fort::separator;
        table << "FRAMEWORK OVERHEAD:" << "" << "" << "" << fort::endr;
        table << "  getDumpInfo() calls" << fmt(overhead.get_dump_info_ms, 2) << per_tok(overhead.get_dump_info_ms) << pct(overhead.get_dump_info_ms) << fort::endr;
        table << "  extractBuffers() calls" << fmt(overhead.extract_buffers_ms, 2) << per_tok(overhead.extract_buffers_ms) << pct(overhead.extract_buffers_ms) << fort::endr;
        table << "  Buffer Verification" << fmt(overhead.verify_ms, 2) << per_tok(overhead.verify_ms) << pct(overhead.verify_ms) << fort::endr;
        table << "  Snapshot Callbacks" << fmt(overhead.callback_ms, 2) << per_tok(overhead.callback_ms) << pct(overhead.callback_ms) << fort::endr;

        // Stage dump
        table << fort::separator;
        table << "STAGE DUMP (if enabled):" << "" << "" << "" << fort::endr;
        table << "  Dump Inputs" << fmt(overhead.dump_input_ms, 2) << per_tok(overhead.dump_input_ms) << pct(overhead.dump_input_ms) << fort::endr;
        table << "  Dump Outputs" << fmt(overhead.dump_output_ms, 2) << per_tok(overhead.dump_output_ms) << pct(overhead.dump_output_ms) << fort::endr;

        // Totals
        table << fort::separator;
        table << "TOTAL OVERHEAD" << fmt(total_overhead, 2) << per_tok(total_overhead) << pct(total_overhead) << fort::endr;
        table << "TOTAL (kernel + overhead)" << fmt(total_all, 2) << per_tok(total_all) << pct(total_all) << fort::endr;

        // Efficiency row
        table << fort::separator;
        double efficiency = total_all > 0 ? (total_execute_ms / total_all) * 100.0 : 0;
        {
            std::ostringstream oss;
            oss << "Kernel Efficiency: " << fmt(efficiency, 1) << "%  (higher = less overhead)";
            if (decode_tokens > 0)
                oss << "  |  Overhead per token: " << fmt(overhead_per_token, 3) << " ms";
            table << oss.str() << "" << "" << "" << fort::endr;
            table[table.row_count() - 1][0].set_cell_span(4);
        }

        std::cout << "\n"
                  << table.to_string() << std::endl;
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
        order_dirty_ = true;
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
        order_dirty_ = true;
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

        order_dirty_ = true;
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
        const auto &order = graph.getExecutionOrder();

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

    bool GraphExecutor::executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                                          const std::unordered_set<std::string> *collective_nodes)
    {
        // Set HIP device once for the entire decode pass — eliminates 339+ redundant hipSetDevice calls
#ifdef HAVE_ROCM
        if (ctx && ctx->deviceId().type == DeviceType::ROCm)
        {
            HipDeviceGuard::forceSetDevice(ctx->deviceId().toKernelDeviceIndex());
        }
#endif

        const auto &order = graph.getExecutionOrder();

        bool collective_graph = (collective_nodes && !collective_nodes->empty());
        if (!collective_graph)
        {
            for (const auto &name : order)
            {
                auto *node = graph.getNode(name);
                if (!node || !node->stage)
                {
                    continue;
                }
                const auto t = node->stage->type();
                if (t == ComputeStageType::ALLREDUCE ||
                    t == ComputeStageType::ALLGATHER ||
                    t == ComputeStageType::ALLGATHER_V)
                {
                    collective_graph = true;
                    break;
                }
            }
        }

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);

            if (!node || !node->stage)
            {
                LOG_ERROR("[GraphExecutor] Fast decode node missing stage: " << name);
                return false;
            }

            const auto stage_type = node->stage->type();
            const bool is_collective_type =
                (stage_type == ComputeStageType::ALLREDUCE ||
                 stage_type == ComputeStageType::ALLGATHER ||
                 stage_type == ComputeStageType::ALLGATHER_V);
            const bool is_collective_node =
                (collective_nodes ? collective_nodes->count(name) > 0 : is_collective_type);

            // Collective-aware handling for TP collectives
            if (is_collective_node)
            {
                if (collective_ctx_ && stage_type == ComputeStageType::ALLREDUCE)
                {
                    if (!executeCollectiveAllreduce(*node, ctx))
                    {
                        LOG_ERROR("[GraphExecutor] Fast decode collective ALLREDUCE failed: " << name);
                        return false;
                    }
                    graph.markCompleted(name);
                    continue;
                }

                if (collective_ctx_ && stage_type == ComputeStageType::ALLGATHER)
                {
                    if (executeCollectiveStridedAllgather(*node, ctx))
                    {
                        graph.markCompleted(name);
                        continue;
                    }
                }

                // Collective-aware safe fallback:
                // Route collective stages through executeNode() so they use the normal
                // coherence/intercept path instead of raw stage->execute().
                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[GraphExecutor] Fast decode collective stage fallback failed: " << name);
                    return false;
                }
                graph.markCompleted(name);
                continue;
            }

            if (collective_graph)
            {
                // Collective graph safety mode:
                // keep fast topological traversal, but execute nodes through the
                // normal node path to preserve coherence/intercept semantics.
                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[GraphExecutor] Fast decode collective-graph node failed: " << name);
                    return false;
                }
                graph.markCompleted(name);
                continue;
            }

            // Non-collective graph: maximal fast path
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[GraphExecutor] Fast decode stage failed: " << name);
                return false;
            }

            graph.markCompleted(name);
        }

        return true;
    }

    bool GraphExecutor::executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                IGPUGraphCapture *capture,
                                                const std::unordered_set<std::string> *collective_nodes,
                                                void *gpu_stream)
    {
        if (!capture)
        {
            LOG_WARN("[GraphExecutor] GPU graph capture is null, falling back to fast decode");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // TP>1 with collectives cannot be captured in a single-device graph
        if (collective_nodes && !collective_nodes->empty())
        {
            LOG_DEBUG("[GraphExecutor] Collective nodes present, skipping graph capture for TP>1");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // Step 1: Begin capture
        if (!capture->beginCapture())
        {
            LOG_WARN("[GraphExecutor] GPU graph beginCapture failed, falling back to fast decode");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // Step 2: Execute all stages into the captured stream
        // Set HIP device once before the loop (same as executeFastDecode)
#ifdef HAVE_ROCM
        if (ctx && ctx->deviceId().type == DeviceType::ROCm)
        {
            HipDeviceGuard::forceSetDevice(ctx->deviceId().toKernelDeviceIndex());
        }
#endif

        const auto &order = graph.getExecutionOrder();
        bool exec_success = true;

        // Set GPU stream on all stages so kernels dispatch to the capture stream
        if (gpu_stream)
        {
            for (const auto &name : order)
            {
                auto *node = graph.getNode(name);
                if (node && node->stage)
                    node->stage->setGPUStream(gpu_stream);
            }
        }

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[GraphExecutor] Stage failed during graph capture: " << name);
                exec_success = false;
                break;
            }
            graph.markCompleted(name);
        }

        // Step 3: End capture
        if (!exec_success || !capture->endCapture())
        {
            LOG_WARN("[GraphExecutor] GPU graph capture failed (exec_success=" << exec_success
                                                                               << "), stream may be in bad state");
            // If capture was started but execute failed, we still need to end capture
            // to restore the stream to a usable state
            if (exec_success)
            {
                // endCapture failed
                capture->reset();
            }
            // Fall through — the stream should be usable again after endCapture
            // The kernels were recorded during capture but NOT executed
            return exec_success;
        }

        // If graph captured 0 nodes, all work was CPU-side (already executed
        // during capture).  Skip instantiate/update/launch — there is nothing
        // to replay.
        if (capture->nodeCount() == 0)
        {
            LOG_WARN("[GraphExecutor] GPU graph captured 0 nodes — kernels NOT on capture stream! Skipping graph replay");
            return true;
        }

        // Step 4: Instantiate or update + launch
        LOG_WARN("[GraphExecutor] GPU graph captured " << capture->nodeCount()
                                                       << " nodes, hasExecutable=" << capture->hasExecutable());
        if (capture->hasExecutable())
        {
            // Try in-place update
            GraphUpdateResult result = capture->tryUpdate();
            LOG_WARN("[GraphExecutor] tryUpdate result=" << static_cast<int>(result));
            if (result == GraphUpdateResult::Success)
            {
                // In-place update succeeded — launch the updated executable
                if (!capture->launch())
                {
                    LOG_ERROR("[GraphExecutor] GPU graph launch failed after update");
                    return false;
                }
                LOG_TRACE("[GraphExecutor] GPU graph updated and launched ("
                          << capture->nodeCount() << " nodes)");
                return true;
            }
            else if (result == GraphUpdateResult::NeedsReinstantiate)
            {
                // Topology changed — reinstantiate
                LOG_WARN("[GraphExecutor] NeedsReinstantiate — calling instantiate()");
                if (!capture->instantiate())
                {
                    LOG_WARN("[GraphExecutor] GPU graph reinstantiation failed");
                    return false;
                }
            }
            else
            {
                // Update failed
                LOG_WARN("[GraphExecutor] GPU graph update failed (result="
                         << static_cast<int>(result) << ")");
                return false;
            }
        }
        else
        {
            // First time — instantiate from captured graph
            LOG_WARN("[GraphExecutor] First instantiation attempt (" << capture->nodeCount() << " nodes)");
            if (!capture->instantiate())
            {
                LOG_WARN("[GraphExecutor] GPU graph instantiation failed");
                return false;
            }
            LOG_WARN("[GraphExecutor] GPU graph instantiated with " << capture->nodeCount()
                                                                    << " nodes (" << capture->backendName() << ")");
        }

        // Launch the (newly instantiated) executable
        if (!capture->launch())
        {
            LOG_ERROR("[GraphExecutor] GPU graph launch failed");
            return false;
        }

        return true;
    }

    bool GraphExecutor::executeWithSegmentedGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                         GraphSegmentCache &segment_cache,
                                                         void *gpu_stream,
                                                         IWorkerGPUContext *gpu_ctx,
                                                         const std::unordered_set<std::string> *collective_nodes)
    {
        if (!gpu_stream || !gpu_ctx)
        {
            LOG_WARN("[GraphExecutor] Segmented graph capture: missing stream or gpu_ctx, falling back");
            return executeFastDecode(graph, ctx);
        }

        const auto &order = graph.getExecutionOrder();
        const bool has_collective_nodes = (collective_nodes && !collective_nodes->empty());

        auto is_collective_stage = [](ComputeStageType t)
        {
            return t == ComputeStageType::ALLREDUCE ||
                   t == ComputeStageType::ALLGATHER ||
                   t == ComputeStageType::ALLGATHER_V;
        };

        auto mark_segment_outputs_dirty = [&](const GraphSegment &seg, void *stream)
        {
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    continue;
                }

                const auto &dump_info = node->stage->getDumpInfo();
                auto outputs = extractOutputBuffers(dump_info);
                markOutputsDirty(outputs, stream);
            }
        };

        auto cohere_segment_inputs = [&](const GraphSegment &seg) -> bool
        {
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    continue;
                }

                const auto policy = node->stage->coherencePolicy();
                if (policy != CoherencePolicy::INPUT && policy != CoherencePolicy::FULL)
                {
                    continue;
                }

                DeviceId target_device = node->device.is_valid() ? node->device : node->stage->device();
                const auto &dump_info = node->stage->getDumpInfo();

                auto inputs = extractInputBuffers(dump_info);
                if (!cohereInputs(inputs, target_device))
                {
                    LOG_ERROR("[GraphExecutor] Failed to cohere replay inputs for stage: " << stage_name);
                    return false;
                }

                auto weights = extractWeightBuffers(dump_info);
                if (!cohereInputs(weights, target_device))
                {
                    LOG_ERROR("[GraphExecutor] Failed to cohere replay weights for stage: " << stage_name);
                    return false;
                }

                if (policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL)
                {
                    auto outputs = extractOutputBuffers(dump_info);
                    if (!cohereOutputs(outputs, target_device))
                    {
                        LOG_ERROR("[GraphExecutor] Failed to cohere replay outputs for stage: " << stage_name);
                        return false;
                    }
                }
            }

            return true;
        };

#ifdef HAVE_ROCM
        if (ctx && ctx->deviceId().type == DeviceType::ROCm)
        {
            HipDeviceGuard::forceSetDevice(ctx->deviceId().toKernelDeviceIndex());
        }
#endif

        // ===== Phase 1: Warmup (first call) — build segments, execute normally =====
        // We do NOT capture on the first call. Some kernels lazily initialize workspace
        // buffers (hipMalloc), which isn't compatible with stream capture.
        // First call builds the segment list and runs via executeFastDecode.
        if (!segment_cache.initialized)
        {
            segment_cache.segments.clear();
            static const std::vector<std::string> segmented_collective_capture_allow = []()
            {
                std::vector<std::string> out;
                const char *raw = std::getenv("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED_CAPTURE_ALLOW");
                if (!raw || !*raw)
                {
                    return out;
                }

                std::stringstream ss(raw);
                std::string token;
                while (std::getline(ss, token, ','))
                {
                    token.erase(0, token.find_first_not_of(" \t"));
                    token.erase(token.find_last_not_of(" \t") + 1);
                    if (!token.empty())
                    {
                        out.push_back(token);
                    }
                }
                return out;
            }();

            auto stage_in_collective_allowlist = [&](const std::string &stage_name) -> bool
            {
                if (segmented_collective_capture_allow.empty())
                {
                    return false;
                }
                for (const auto &needle : segmented_collective_capture_allow)
                {
                    if (stage_name.find(needle) != std::string::npos)
                    {
                        return true;
                    }
                }
                return false;
            };

            // Partition execution order into contiguous runs of capturable / non-capturable stages.
            // Collective stages are always treated as non-capturable boundaries in phase 2.
            bool current_capturable = false;
            bool first = true;

            for (const auto &name : order)
            {
                auto *node = graph.getNode(name);
                if (!node || !node->stage)
                    continue;

                bool stage_capturable = node->stage->isGraphCapturable();
                const bool collective_by_type = is_collective_stage(node->stage->type());
                const bool collective_by_name = (collective_nodes && collective_nodes->count(name));
                if (collective_by_type || collective_by_name)
                {
                    stage_capturable = false;
                }

                // Optional narrowing/debug override for collective segmented mode:
                // when set, ONLY stages whose names match this comma-separated
                // allowlist are capturable.
                //   LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED_CAPTURE_ALLOW=a,b,c
                if (has_collective_nodes && !segmented_collective_capture_allow.empty())
                {
                    stage_capturable = stage_in_collective_allowlist(name);
                }

                // Conservative safety for collective segmented replay:
                // dynamic-parameter stages (e.g., attention/RoPE) mutate per-step
                // device params and are currently unsafe to replay-capture here.
                if (has_collective_nodes && node->stage->hasDynamicParams())
                {
                    stage_capturable = false;
                }

                // Root-cause guard: residual add stages are non-idempotent in
                // segmented collective replay and can corrupt decode output.
                if (has_collective_nodes && node->stage->type() == ComputeStageType::ADD_RESIDUAL)
                {
                    stage_capturable = false;
                }

                // Additional conservative safety for collective segmented replay:
                // keep final output path manual so logits/sampling boundary always
                // runs via executeNode() coherence semantics.
                if (has_collective_nodes && (name == "final_norm" || name == "lm_head"))
                {
                    stage_capturable = false;
                }

                // Start a new segment if capturability changes or this is the first stage
                if (first || stage_capturable != current_capturable)
                {
                    segment_cache.segments.emplace_back();
                    segment_cache.segments.back().capturable = stage_capturable;
                    current_capturable = stage_capturable;
                    first = false;
                }

                segment_cache.segments.back().stage_names.push_back(name);
            }

            // Split oversized capturable segments if max_stages is set
            const int max_stages = debugEnv().execution.gpu_graph_max_stages;
            if (max_stages > 0)
            {
                std::vector<GraphSegment> split_segments;
                for (auto &seg : segment_cache.segments)
                {
                    if (seg.capturable && static_cast<int>(seg.stage_names.size()) > max_stages)
                    {
                        // Split into chunks of max_stages
                        for (size_t i = 0; i < seg.stage_names.size(); i += max_stages)
                        {
                            GraphSegment sub;
                            sub.capturable = true;
                            size_t end = std::min(i + max_stages, seg.stage_names.size());
                            for (size_t j = i; j < end; j++)
                                sub.stage_names.push_back(seg.stage_names[j]);
                            split_segments.push_back(std::move(sub));
                        }
                    }
                    else
                    {
                        split_segments.push_back(std::move(seg));
                    }
                }
                segment_cache.segments = std::move(split_segments);
            }

            // Log segment layout
            size_t capturable_segments = 0, manual_segments = 0;
            size_t capturable_stages = 0, manual_stages = 0;
            for (const auto &seg : segment_cache.segments)
            {
                if (seg.capturable)
                {
                    capturable_segments++;
                    capturable_stages += seg.stage_names.size();
                }
                else
                {
                    manual_segments++;
                    manual_stages += seg.stage_names.size();
                }
            }

            LOG_INFO("[GraphExecutor] Segmented graph: " << capturable_segments << " capturable segments ("
                                                         << capturable_stages << " stages) + " << manual_segments << " manual segments ("
                                                         << manual_stages << " stages)");

            // Warmup: execute all stages normally (no capture) to ensure
            // lazy kernel initialization and workspace allocation complete.
            segment_cache.initialized = true;
            segment_cache.needs_capture = true;
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // ===== Phase 2: Capture (second call) — record capturable segments =====
        if (segment_cache.needs_capture)
        {
            segment_cache.needs_capture = false;

            // Create a local blocking stream for capture/replay.
            // CRITICAL: We create this stream directly on the calling thread, NOT
            // from the AMDDeviceContext worker thread. On ROCm (MI50/gfx906, ROCm 7.1),
            // streams created on a different thread produce corrupted output when used
            // for graph capture on the main thread. Creating the stream locally avoids
            // this cross-thread issue.
            if (!segment_cache.ensureCaptureStream(gpu_ctx))
            {
                LOG_WARN("[GraphExecutor] Failed to create capture stream, falling back");
                return executeFastDecode(graph, ctx, collective_nodes);
            }
            void *capture_stream = segment_cache.capture_stream;

            for (auto &seg : segment_cache.segments)
            {
                if (seg.capturable)
                {
                    // Set the local capture stream on stages in this segment
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (node && node->stage)
                            node->stage->setGPUStream(capture_stream);
                    }

                    // Create graph capture using our local stream via factory (avoids HIP/CUDA header collision)
                    seg.capture = gpu_ctx->createGraphCapture(capture_stream);
                    if (!seg.capture)
                    {
                        LOG_ERROR("[GraphExecutor] Failed to create graph capture for segment");
                        segment_cache.reset();
                        return executeFastDecode(graph, ctx);
                    }

                    // Begin capture
                    if (!seg.capture->beginCapture())
                    {
                        LOG_ERROR("[GraphExecutor] beginCapture failed for segment");
                        segment_cache.reset();
                        return executeFastDecode(graph, ctx);
                    }

                    // Execute all stages in this segment into the capture stream
                    bool exec_ok = true;
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node->stage->execute(ctx))
                        {
                            LOG_ERROR("[GraphExecutor] Stage failed during segmented capture: " << stage_name);
                            exec_ok = false;
                            break;
                        }
                        graph.markCompleted(stage_name);
                    }

                    // End capture
                    if (!exec_ok || !seg.capture->endCapture())
                    {
                        LOG_ERROR("[GraphExecutor] Segmented capture failed");
                        segment_cache.reset();
                        return exec_ok;
                    }

                    // Instantiate and launch immediately so subsequent segments have valid data
                    if (seg.capture->nodeCount() > 0)
                    {
                        if (!seg.capture->instantiate())
                        {
                            LOG_WARN("[GraphExecutor] Segment instantiation failed ("
                                     << seg.capture->nodeCount() << " nodes)");
                            segment_cache.reset();
                            return false;
                        }
                        if (!seg.capture->launch())
                        {
                            LOG_ERROR("[GraphExecutor] Segment initial launch failed");
                            segment_cache.reset();
                            return false;
                        }
                        mark_segment_outputs_dirty(seg, capture_stream);
                        // Sync before moving to next segment
                        gpu_ctx->synchronize();
                        LOG_DEBUG("[GraphExecutor] Segment captured+launched: " << seg.capture->nodeCount()
                                                                                << " nodes, " << seg.stage_names.size() << " stages");
                    }
                    else
                    {
                        LOG_DEBUG("[GraphExecutor] Segment captured 0 nodes (CPU-only), will execute manually");
                        seg.capture.reset();
                    }
                }
                else if (!seg.capturable)
                {
                    // Manual segment — execute stages on default stream (nullptr)
                    gpu_ctx->synchronize(); // Sync prior graph segment

                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage)
                        {
                            LOG_ERROR("[GraphExecutor] Manual segment missing stage: " << stage_name);
                            segment_cache.reset();
                            return false;
                        }

                        node->stage->setGPUStream(nullptr);
                        if (has_collective_nodes)
                        {
                            if (!executeNode(*node, ctx))
                            {
                                LOG_ERROR("[GraphExecutor] Manual stage failed during capture (collective graph): " << stage_name);
                                segment_cache.reset();
                                return false;
                            }
                        }
                        else if (is_collective_stage(node->stage->type()))
                        {
                            if (!executeNode(*node, ctx))
                            {
                                LOG_ERROR("[GraphExecutor] Manual collective stage failed during capture: " << stage_name);
                                segment_cache.reset();
                                return false;
                            }
                        }
                        else if (!node->stage->execute(ctx))
                        {
                            LOG_ERROR("[GraphExecutor] Manual stage failed: " << stage_name);
                            segment_cache.reset();
                            return false;
                        }
                        graph.markCompleted(stage_name);
                    }

                    // Sync before next graph segment
                    gpu_ctx->synchronize();
                }
            }

            // Precompute replay_callbacks: cache pointers to stages that override
            // onGraphReplayed(), avoiding per-step hash map lookups in Phase 3.
            for (auto &seg : segment_cache.segments)
            {
                if (!seg.capturable)
                    continue;
                seg.replay_callbacks.clear();
                for (const auto &stage_name : seg.stage_names)
                {
                    auto *node = graph.getNode(stage_name);
                    if (node && node->stage && node->stage->needsOnGraphReplayed())
                        seg.replay_callbacks.push_back(node->stage.get());
                }
            }

            return true;
        }

        // ===== Phase 3: Replay — just launch() capturable segments directly =====
        // SYNCHRONIZATION STRATEGY (Unified-Stream):
        // ALL work — both graph launches and manual stages — runs on capture_stream.
        // Since all GPU operations are on the SAME stream, the GPU guarantees
        // in-order execution. NO intermediate CPU-side synchronization is needed.
        //
        // Manual stages' CPU code (metadata reads, mask creation) doesn't depend
        // on GPU results being visible — it only reads CPU-side state that was
        // updated during previous execute() calls (e.g., KV cache entry.count).
        //
        // The ONLY sync point is the final device sync after all segments,
        // ensuring GPU work completes before the caller reads output tensors.

        void *capture_stream = segment_cache.capture_stream;
        const auto &exec_cfg = debugEnv().execution;
        const bool verify_mode = exec_cfg.gpu_graph_verify;
        const bool recapture_mode = exec_cfg.gpu_graph_recapture;
        const bool needs_segment_sync = ctx->deviceId().is_cuda(); // CUDA needs explicit stream sync between segments

        // Stream-only mode: execute all stages on capture_stream WITHOUT graph capture.
        // Tests whether the non-default stream itself causes issues.
        static const bool stream_only_mode = (std::getenv("LLAMINAR_GPU_GRAPH_STREAM_ONLY") &&
                                              std::atoi(std::getenv("LLAMINAR_GPU_GRAPH_STREAM_ONLY")) != 0);
        // Sub-mode: use default stream instead of capture_stream (control test)
        static const bool stream_only_default = (std::getenv("LLAMINAR_GPU_GRAPH_STREAM_ONLY_DEFAULT") &&
                                                 std::atoi(std::getenv("LLAMINAR_GPU_GRAPH_STREAM_ONLY_DEFAULT")) != 0);
        if (stream_only_mode)
        {
            void *use_stream = stream_only_default ? nullptr : capture_stream;
            for (auto &seg : segment_cache.segments)
            {
                for (const auto &stage_name : seg.stage_names)
                {
                    auto *node = graph.getNode(stage_name);
                    node->stage->setGPUStream(use_stream);
                    if (!node->stage->execute(ctx))
                    {
                        LOG_ERROR("[GraphExecutor] Stream-only stage failed: " << stage_name);
                        return false;
                    }
                    graph.markCompleted(stage_name);
                }
            }
            gpu_ctx->synchronize();
            return true;
        }

        int seg_idx = 0;
        for (auto &seg : segment_cache.segments)
        {
            auto segment_has_non_idempotent_stage = [&]() -> bool
            {
                for (const auto &stage_name : seg.stage_names)
                {
                    auto *node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        continue;
                    }
                    const auto stage_type = node->stage->type();
                    if (stage_type == ComputeStageType::ADD_RESIDUAL)
                    {
                        return true;
                    }
                }
                return false;
            };

            if (seg.capturable && seg.capture && seg.capture->hasExecutable())
            {

                if (!cohere_segment_inputs(seg))
                {
                    return false;
                }

                if (recapture_mode)
                {
                    // ===== Re-capture mode: capture fresh graph each iteration =====
                    // This tests whether re-capturing fixes replay corruption.
                    // Set stages to capture stream
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (node && node->stage)
                            node->stage->setGPUStream(capture_stream);
                    }

                    // Re-capture
                    if (!seg.capture->beginCapture())
                    {
                        LOG_ERROR("[GraphExecutor] Re-capture beginCapture failed, seg " << seg_idx);
                        return false;
                    }

                    bool exec_ok = true;
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node->stage->execute(ctx))
                        {
                            exec_ok = false;
                            break;
                        }
                    }

                    if (!exec_ok || !seg.capture->endCapture())
                    {
                        LOG_ERROR("[GraphExecutor] Re-capture failed, seg " << seg_idx);
                        return false;
                    }

                    // Try in-place update, else reinstantiate
                    auto update_result = seg.capture->tryUpdate();
                    if (update_result == GraphUpdateResult::NeedsReinstantiate)
                    {
                        if (!seg.capture->instantiate())
                        {
                            LOG_ERROR("[GraphExecutor] Re-capture reinstantiate failed");
                            return false;
                        }
                    }
                    else if (update_result == GraphUpdateResult::Failed)
                    {
                        if (!seg.capture->instantiate())
                        {
                            LOG_ERROR("[GraphExecutor] Re-capture instantiate failed");
                            return false;
                        }
                    }

                    // Launch the updated graph
                    if (!seg.capture->launch())
                    {
                        LOG_ERROR("[GraphExecutor] Re-capture launch failed");
                        return false;
                    }

                    mark_segment_outputs_dirty(seg, capture_stream);

                    gpu_ctx->synchronize();

                    for (const auto &stage_name : seg.stage_names)
                        graph.markCompleted(stage_name);
                }
                else if (verify_mode)
                {
                    if (segment_has_non_idempotent_stage())
                    {
                        if (!seg.capture->launch())
                        {
                            LOG_ERROR("[GraphExecutor] Verify-skip: graph launch failed, seg " << seg_idx);
                            return false;
                        }

                        mark_segment_outputs_dirty(seg, capture_stream);
                        for (auto *stage : seg.replay_callbacks)
                            stage->onGraphReplayed();

                        if (needs_segment_sync)
                        {
                            gpu_ctx->synchronizeStream(capture_stream);
                        }

                        fprintf(stderr,
                                "[GRAPH_VERIFY] seg %d skipped (non-idempotent stage detected)\n",
                                seg_idx);

                        for (const auto &stage_name : seg.stage_names)
                            graph.markCompleted(stage_name);

                        seg_idx++;
                        continue;
                    }

                    // ===== Verify mode: per-stage graph replay vs direct exec comparison =====
                    // Step 1: Launch graph replay (executes ALL stages in segment)
                    if (!seg.capture->launch())
                    {
                        LOG_ERROR("[GraphExecutor] Verify: graph launch failed, seg " << seg_idx);
                        return false;
                    }
                    mark_segment_outputs_dirty(seg, capture_stream);
                    for (auto *stage : seg.replay_callbacks)
                        stage->onGraphReplayed();
                    gpu_ctx->synchronize();

                    // Step 2: Read output of EVERY stage after graph replay
                    struct StageOutput
                    {
                        std::string name;
                        float values[8] = {};
                        size_t count = 0;
                        bool has_gpu_ptr = false;
                    };
                    std::vector<StageOutput> graph_outputs(seg.stage_names.size());

                    for (size_t s = 0; s < seg.stage_names.size(); s++)
                    {
                        auto *node = graph.getNode(seg.stage_names[s]);
                        graph_outputs[s].name = seg.stage_names[s];
                        if (node && node->stage)
                        {
                            const auto &dump_info = node->stage->getDumpInfo();
                            if (!dump_info.outputs.empty())
                            {
                                const auto &out = dump_info.outputs[0];
                                if (out.tensor)
                                {
                                    auto *base = dynamic_cast<TensorBase *>(out.tensor);
                                    if (base)
                                    {
                                        const void *gpu_ptr = base->gpu_data_ptr();
                                        if (gpu_ptr)
                                        {
                                            graph_outputs[s].count = std::min<size_t>(8, out.rows * out.cols);
                                            graph_outputs[s].has_gpu_ptr =
                                                ctx->copyToHost(graph_outputs[s].values, gpu_ptr,
                                                                graph_outputs[s].count * sizeof(float));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Step 3: Re-execute ALL stages directly (overwrites graph output)
                    for (const auto &stage_name : seg.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        node->stage->setGPUStream(nullptr);
                        if (!node->stage->execute(ctx))
                        {
                            LOG_ERROR("[GraphExecutor] Verify: direct exec failed: " << stage_name);
                            return false;
                        }
                    }
                    gpu_ctx->synchronize();

                    // Step 4: Read output of EVERY stage after direct execution
                    std::vector<StageOutput> direct_outputs(seg.stage_names.size());
                    for (size_t s = 0; s < seg.stage_names.size(); s++)
                    {
                        auto *node = graph.getNode(seg.stage_names[s]);
                        direct_outputs[s].name = seg.stage_names[s];
                        if (node && node->stage)
                        {
                            const auto &dump_info = node->stage->getDumpInfo();
                            if (!dump_info.outputs.empty())
                            {
                                const auto &out = dump_info.outputs[0];
                                if (out.tensor)
                                {
                                    auto *base = dynamic_cast<TensorBase *>(out.tensor);
                                    if (base)
                                    {
                                        const void *gpu_ptr = base->gpu_data_ptr();
                                        if (gpu_ptr)
                                        {
                                            direct_outputs[s].count = std::min<size_t>(8, out.rows * out.cols);
                                            direct_outputs[s].has_gpu_ptr =
                                                ctx->copyToHost(direct_outputs[s].values, gpu_ptr,
                                                                direct_outputs[s].count * sizeof(float));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Step 5: Per-stage comparison and logging
                    FILE *f = fopen("/tmp/graph_phase3.log", "a");
                    float seg_max_diff = 0;

                    for (size_t s = 0; s < seg.stage_names.size(); s++)
                    {
                        if (!graph_outputs[s].has_gpu_ptr || !direct_outputs[s].has_gpu_ptr)
                            continue;

                        float max_diff = 0;
                        size_t n = std::min(graph_outputs[s].count, direct_outputs[s].count);
                        for (size_t i = 0; i < n; i++)
                            max_diff = std::max(max_diff, std::abs(graph_outputs[s].values[i] - direct_outputs[s].values[i]));
                        seg_max_diff = std::max(seg_max_diff, max_diff);

                        char buf[1024];
                        snprintf(buf, sizeof(buf),
                                 "[STAGE_VERIFY] seg %d stage %zu/%zu (%s) max_diff=%.6e",
                                 seg_idx, s, seg.stage_names.size(), seg.stage_names[s].c_str(), max_diff);

                        if (f)
                        {
                            fprintf(f, "%s\n", buf);
                            if (max_diff > 1e-6f)
                            {
                                fprintf(f, "  GRAPH : ");
                                for (size_t i = 0; i < n; i++)
                                    fprintf(f, "%.6f%s", graph_outputs[s].values[i], i < n - 1 ? ", " : "");
                                fprintf(f, "\n  DIRECT: ");
                                for (size_t i = 0; i < n; i++)
                                    fprintf(f, "%.6f%s", direct_outputs[s].values[i], i < n - 1 ? ", " : "");
                                fprintf(f, "\n");
                            }
                            fflush(f);
                        }
                    }

                    // Summary line
                    {
                        char buf[512];
                        snprintf(buf, sizeof(buf),
                                 "[GRAPH_VERIFY] seg %d (%zu stages, %zu nodes, last=%s) seg_max_diff=%.6e",
                                 seg_idx, seg.stage_names.size(), seg.capture->nodeCount(),
                                 seg.stage_names.back().c_str(), seg_max_diff);
                        fprintf(stderr, "%s\n", buf);
                        if (f)
                        {
                            fprintf(f, "%s\n\n", buf);
                            fflush(f);
                        }
                    }

                    if (f)
                        fclose(f);

                    for (const auto &stage_name : seg.stage_names)
                        graph.markCompleted(stage_name);
                }
                else
                {
                    // ===== Normal replay mode =====
                    if (!seg.capture->launch())
                    {
                        LOG_ERROR("[GraphExecutor] Segment graph launch failed on replay");
                        segment_cache.consecutive_failures++;
                        if (segment_cache.consecutive_failures >= GraphSegmentCache::kMaxFailures)
                        {
                            LOG_WARN("[GraphExecutor] Too many segmented graph failures, disabling");
                            segment_cache.reset();
                        }
                        graph.reset();
                        return executeFastDecode(graph, ctx, collective_nodes);
                    }

                    // Captured segments bypass executeNode(), so explicitly mark
                    // outputs device-authoritative before any manual/coherent stage.
                    mark_segment_outputs_dirty(seg, capture_stream);

                    // Run post-replay hooks via precomputed list (no hash lookups)
                    for (auto *stage : seg.replay_callbacks)
                        stage->onGraphReplayed();

                    // NOTE: markCompleted is intentionally skipped for capturable
                    // segments — graph.reset() clears all flags at the start of
                    // the next step, so marking is unnecessary overhead.

                    // CUDA workaround: Unlike ROCm/HIP graphs which properly
                    // order graph launches w.r.t. subsequent kernel launches on the
                    // same stream, CUDA requires an explicit stream sync between
                    // the last manual segment and the next graph launch. Without this,
                    // the replayed graph may read stale data from manual stage outputs.
                    //
                    // Use stream-level sync (not device-wide) for minimal overhead.
                    // ROCm doesn't need this — stream ordering alone works.
                    if (needs_segment_sync)
                    {
                        gpu_ctx->synchronizeStream(capture_stream);
                    }
                }
            }
            else
            {
                // Manual segment — execute on capture_stream (unified with graph segments)
                bool manual_had_collective = false;
                for (const auto &stage_name : seg.stage_names)
                {
                    auto *node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        LOG_ERROR("[GraphExecutor] Manual segment missing stage: " << stage_name);
                        return false;
                    }

                    const auto t = node->stage->type();
                    if (has_collective_nodes)
                    {
                        manual_had_collective = manual_had_collective || is_collective_stage(t);

                        if (needs_segment_sync)
                        {
                            gpu_ctx->synchronizeStream(capture_stream);
                        }

                        node->stage->setGPUStream(nullptr);
                        if (!executeNode(*node, ctx))
                        {
                            LOG_ERROR("[GraphExecutor] Manual stage failed on replay (collective graph): " << stage_name);
                            return false;
                        }

                        gpu_ctx->synchronize();
                    }
                    else if (is_collective_stage(t))
                    {
                        // Collective manual stages must go through executeNode() so
                        // intercept/coherence behavior matches non-segmented execution.
                        manual_had_collective = true;

                        // Ensure all prior capture-stream work is visible before
                        // running a collective stage that may use a different stream.
                        if (needs_segment_sync)
                        {
                            gpu_ctx->synchronizeStream(capture_stream);
                        }

                        node->stage->setGPUStream(nullptr);
                        if (!executeNode(*node, ctx))
                        {
                            LOG_ERROR("[GraphExecutor] Manual collective stage failed on replay: " << stage_name);
                            return false;
                        }

                        // Collective backends may execute on non-capture streams;
                        // enforce completion before subsequent capture-stream stages.
                        gpu_ctx->synchronize();
                    }
                    else
                    {
                        node->stage->setGPUStream(capture_stream);
                        if (!node->stage->execute(ctx))
                        {
                            LOG_ERROR("[GraphExecutor] Manual stage failed on replay: " << stage_name);
                            return false;
                        }
                    }
                    // markCompleted skipped — graph.reset() clears at next step
                }
                // CUDA workaround: sync stream after manual segment before next graph
                if (needs_segment_sync)
                {
                    if (manual_had_collective)
                    {
                        gpu_ctx->synchronize();
                    }
                    else
                    {
                        gpu_ctx->synchronizeStream(capture_stream);
                    }
                }
            }

            seg_idx++;
        }

        // Final safety sync for segmented replay:
        // segmented capture uses a dedicated local capture stream managed by gpu_ctx,
        // which may not be covered by caller-side ctx->synchronize() semantics.
        // Ensure all capture/default-stream work is complete before returning.
        gpu_ctx->synchronize();

        segment_cache.consecutive_failures = 0;
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
        const auto &order = graph.getExecutionOrder();

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
                // In segmented-collective graph mode, prefer stage execution path
                // to preserve the same coherence/intercept behavior as baseline.
                if (debugEnv().execution.gpu_graph_collective_segmented)
                {
                    LOG_DEBUG("[GraphExecutor] Skipping strided ALLGATHER intercept in segmented collective mode for '" << node.name << "'");
                }
                else
                {
                    LOG_DEBUG("[GraphExecutor] Attempting strided ALLGATHER intercept for '" << node.name << "'");
                    if (executeCollectiveStridedAllgather(node, ctx))
                    {
                        return true;
                    }
                    // Fall through to normal execution if strided path not available
                    LOG_DEBUG("[GraphExecutor] Strided ALLGATHER not available, using stage execution");
                }
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

                markOutputsDirty(outputs, node.stage->gpuStream());
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
