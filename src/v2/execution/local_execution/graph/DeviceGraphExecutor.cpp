/**
 * @file DeviceGraphExecutor.cpp
 * @brief Compute graph execution engine implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "DeviceGraphExecutor.h"
#include "DeviceGraphCaptureController.h"
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
#include "../../../backends/GPUDeviceContextPool.h"
#ifdef HAVE_ROCM
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../backends/rocm/ROCmBackend.h"
#endif
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <print>
#include "fort.hpp"
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <cstdint>

namespace llaminar2
{
    namespace
    {
        void *resolveWorkerDefaultStream(const DeviceId &device)
        {
            auto &pool = GPUDeviceContextPool::instance();

            try
            {
#ifdef HAVE_ROCM
                if (device.is_rocm() && pool.hasAMDSupport() &&
                    device.rocm_ordinal() >= 0 &&
                    device.rocm_ordinal() < pool.amdDeviceCount())
                {
                    return pool.getAMDContext(device.rocm_ordinal()).defaultStream();
                }
#endif

#ifdef HAVE_CUDA
                if (device.is_cuda() && pool.hasNvidiaSupport() &&
                    device.cuda_ordinal() >= 0 &&
                    device.cuda_ordinal() < pool.nvidiaDeviceCount())
                {
                    return pool.getNvidiaContext(device.cuda_ordinal()).defaultStream();
                }
#endif
            }
            catch (const std::exception &e)
            {
                LOG_DEBUG("[DeviceGraphExecutor] Failed to resolve worker default stream for "
                          << device.to_string() << ": " << e.what());
            }

            return nullptr;
        }

        void ensureStageGPUStreamBound(ComputeNode &node, IDeviceContext *ctx)
        {
            if (!node.stage || node.stage->gpuStream() != nullptr)
            {
                return;
            }

            DeviceId device = node.device.is_valid() ? node.device : node.stage->device();
            if (!device.is_valid() && ctx)
            {
                device = ctx->deviceId();
            }

            void *stream = resolveWorkerDefaultStream(device);
            if (stream)
            {
                node.stage->setGPUStream(stream);
            }
        }
    }

    // =========================================================================
    // GraphSegmentCache — capture stream management
    // =========================================================================

    bool DeviceGraphExecutor::GraphSegmentCache::ensureCaptureStream(IWorkerGPUContext *ctx)
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

    void DeviceGraphExecutor::GraphSegmentCache::destroyCaptureStream()
    {
        if (!capture_stream)
            return;
        if (gpu_ctx_ref)
            gpu_ctx_ref->destroyStream(capture_stream);
        capture_stream = nullptr;
        gpu_ctx_ref = nullptr;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureSyncEvent(IWorkerGPUContext *ctx)
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

    void DeviceGraphExecutor::GraphSegmentCache::destroySyncEvent()
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

        std::print("\n{}", table.to_string());
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
    // DeviceGraphExecutor Implementation
    // =============================================================================

    DeviceGraphExecutor::DeviceGraphExecutor(const GraphExecutorConfig &config)
        : config_(config) {}

    DeviceGraphExecutor::~DeviceGraphExecutor() = default;

    // =============================================================================
    // Execution
    // =============================================================================

    bool DeviceGraphExecutor::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphExecutor] Null device context");
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
            LOG_WARN("[DeviceGraphExecutor] Pipelined mode not yet implemented, using sequential");
            return executeSequential(graph, ctx);
        default:
            LOG_ERROR("[DeviceGraphExecutor] Unknown execution mode");
            return false;
        }
    }

    bool DeviceGraphExecutor::executeSequential(ComputeGraph &graph, IDeviceContext *ctx)
    {
        // Set HIP device for this thread (critical for multi-GPU LocalTP prefill)
        // Without this, std::async threads may not have the correct HIP device context,
        // causing cross-device memory access faults when coherence or kernels allocate memory.
        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

        const auto &order = graph.getExecutionOrder();

        // =====================================================================
        // Mark the last stage as needing event-based dirty marking.
        // Its outputs will be read back to CPU (for sampling, verification,
        // or snapshot capture). Without an event recorded on the compute stream,
        // ensureOnHost() cannot synchronize properly when the compute stream
        // uses cudaStreamNonBlocking / hipStreamNonBlocking — cudaMemcpy on
        // the NULL stream does NOT implicitly synchronize with non-blocking
        // streams, so D2H would race with the final kernel and copy stale data.
        // =====================================================================
        if (!order.empty())
        {
            auto *last_node = graph.getNode(order.back());
            if (last_node)
                last_node->is_final_output = true;
        }

        // =====================================================================
        // Multi-GPU Stage Sync
        //
        // In LocalTP (multi-device) execution, each device runs its own graph
        // on a separate std::async thread. Without explicit device-wide
        // synchronization between stages, HIP/ROCm issues memory access faults
        // because host-to-device coherence transfers (hipMemcpy on NULL stream)
        // and compute kernels (on AMDDeviceContext::default_stream_) can race.
        //
        // The pre-stage sync ensures all prior GPU work (including RCCL
        // collectives) has completed before the next stage's coherence
        // operations begin. The post-stage sync ensures kernel output is
        // available for subsequent stages or collective reads.
        // =====================================================================
        const bool multi_gpu_sync = collective_ctx_ && ctx->isGPU();
        [[maybe_unused]] const int device_ordinal = ctx->isGPU() ? ctx->deviceId().toKernelDeviceIndex() : -1;

        auto total_start = std::chrono::high_resolution_clock::now();

        int stage_idx = 0;
        auto prev_time = total_start;

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphExecutor] Invalid node: " << name);
                return false;
            }

            ensureStageGPUStreamBound(*node, ctx);

            auto stage_start = std::chrono::high_resolution_clock::now();

            // GPU pointer diagnostics for multi-GPU (no sync needed - just query attributes)
#ifdef HAVE_ROCM
            if (multi_gpu_sync && ctx->deviceId().is_rocm() && debugEnv().validation.validate_gpu_ptrs)
            {
                const StageDumpInfo &dump_info = node->stage->getDumpInfo();
                auto check_ptr = [&](const char *category, const char *tname, ITensor *tensor)
                {
                    if (!tensor)
                        return;
                    auto *tb = dynamic_cast<TensorBase *>(tensor);
                    if (!tb)
                        return;
                    void *gpu_ptr = tb->gpu_data_ptr();
                    if (!gpu_ptr)
                        return;
                    hipPointerAttribute_t attr{};
                    hipError_t err = hipPointerGetAttributes(&attr, gpu_ptr);
                    if (err == hipSuccess && attr.device != device_ordinal)
                    {
                        LOG_ERROR("[GPU_PTR_CHECK] WRONG DEVICE! stage='" << name
                                                                          << "' " << category << " tensor='" << (tname ? tname : "?")
                                                                          << "' gpu_ptr=" << gpu_ptr
                                                                          << " is on device " << attr.device
                                                                          << " but expected device " << device_ordinal);
                    }
                };
                for (const auto &inp : dump_info.inputs)
                    check_ptr("input", inp.name, inp.tensor);
                for (const auto &out : dump_info.outputs)
                    check_ptr("output", out.name, out.tensor);
                for (const auto &w : dump_info.weights)
                    check_ptr("weight", w.name, const_cast<ITensor *>(w.tensor));
            }
#endif

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage failed: " << name);
                return false;
            }

            auto stage_end = std::chrono::high_resolution_clock::now();
            double stage_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();

            // Per-stage timing - TRACE level (use LLAMINAR_EXECUTOR_PROFILING=1 for detailed stats)
            LOG_TRACE("[DeviceGraphExecutor] Stage " << stage_idx << "/" << order.size() << ": " << name << " took " << stage_ms << "ms");
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
                LOG_INFO("[DeviceGraphExecutor] Waiting for " << pending << " pending async dumps...");
                AsyncStageDumper::waitForCompletion();
            }
        }

        LOG_DEBUG("[DeviceGraphExecutor] Total execution: " << total_ms << "ms for " << order.size() << " stages (" << (total_ms / order.size()) << "ms/stage avg)");

        stats_.total_time_ms += total_ms;
        // Only increment here if profiling is disabled - executeNode already increments when profiling is enabled
        if (!config_.enable_profiling)
        {
            stats_.total_stages_executed += order.size();
        }
        stats_.total_flops += graph.totalEstimatedFlops();

        return true;
    }

    bool DeviceGraphExecutor::executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                                                const std::unordered_set<std::string> *collective_nodes)
    {
        // Set HIP device once for the entire decode pass — eliminates 339+ redundant hipSetDevice calls
        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

        const auto &order = graph.getExecutionOrder();
        const bool profile_full_node_path = config_.enable_profiling;

        // =====================================================================
        // Mark the last stage as needing event-based dirty marking (same
        // rationale as executeSequential — see comment there).
        // =====================================================================
        if (!order.empty())
        {
            auto *last_node = graph.getNode(order.back());
            if (last_node)
                last_node->is_final_output = true;
        }

        // Multi-GPU stage sync (same rationale as executeSequential)
        const bool multi_gpu_sync = collective_ctx_ && ctx->isGPU();
        [[maybe_unused]] const int device_ordinal = ctx->isGPU() ? ctx->deviceId().toKernelDeviceIndex() : -1;
        [[maybe_unused]] const bool is_rocm = ctx->deviceId().is_rocm();

        // Helper lambdas for pre/post stage sync
        // With stream-level sync in the RCCL/NCCL coordinators (event-based pre-sync
        // + stream sync post-sync), per-stage device sync is no longer needed.
        // Compute stages run on the same stream (implicit ordering), and the
        // coordinator handles cross-stream sync for collectives.
#ifdef HAVE_ROCM
        auto pre_stage_sync = [&]()
        {
            // No-op: coordinator handles compute→collective sync via stream-wait-event
        };
        auto post_stage_sync = [&]()
        {
            // No-op: coordinator handles collective→compute sync via host-side stream sync
        };
#else
        auto pre_stage_sync = []() {};
        auto post_stage_sync = []() {};
#endif

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
                LOG_ERROR("[DeviceGraphExecutor] Fast decode node missing stage: " << name);
                return false;
            }

            const auto stage_type = node->stage->type();
            const bool is_collective_type =
                (stage_type == ComputeStageType::ALLREDUCE ||
                 stage_type == ComputeStageType::ALLGATHER ||
                 stage_type == ComputeStageType::ALLGATHER_V);
            const bool is_collective_node =
                (collective_nodes ? collective_nodes->count(name) > 0 : is_collective_type);

            // Profiling mode: always route through executeNode() so stage timing,
            // coherence overhead, transfer attribution, and callbacks are captured.
            // Fast-path bypass remains unchanged when profiling is disabled.
            if (profile_full_node_path)
            {
                pre_stage_sync();
                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Fast decode profiled node failed: " << name);
                    return false;
                }
                post_stage_sync();
                graph.markCompleted(name);
                continue;
            }

            // Collective-aware handling for TP collectives
            if (is_collective_node)
            {
                if (collective_ctx_ && stage_type == ComputeStageType::ALLREDUCE)
                {
                    pre_stage_sync();
                    if (!executeCollectiveAllreduce(*node, ctx))
                    {
                        LOG_ERROR("[DeviceGraphExecutor] Fast decode collective ALLREDUCE failed: " << name);
                        return false;
                    }
                    post_stage_sync();
                    graph.markCompleted(name);
                    continue;
                }

                if (collective_ctx_ && stage_type == ComputeStageType::ALLGATHER)
                {
                    pre_stage_sync();
                    if (executeCollectiveStridedAllgather(*node, ctx))
                    {
                        post_stage_sync();
                        graph.markCompleted(name);
                        continue;
                    }
                    post_stage_sync();
                }

                // Collective-aware safe fallback:
                // Route collective stages through executeNode() so they use the normal
                // coherence/intercept path instead of raw stage->execute().
                pre_stage_sync();
                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Fast decode collective stage fallback failed: " << name);
                    return false;
                }
                post_stage_sync();
                graph.markCompleted(name);
                continue;
            }

            if (collective_graph)
            {
                // Collective graph safety mode:
                // keep fast topological traversal, but execute nodes through the
                // normal node path to preserve coherence/intercept semantics.
                pre_stage_sync();
                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Fast decode collective-graph node failed: " << name);
                    return false;
                }
                post_stage_sync();
                graph.markCompleted(name);
                continue;
            }

            // Non-collective graph: maximal fast path
            pre_stage_sync();
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[DeviceGraphExecutor] Fast decode stage failed: " << name);
                return false;
            }
            post_stage_sync();

            graph.markCompleted(name);
        }

        return true;
    }

    bool DeviceGraphExecutor::executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                      IGPUGraphCapture *capture,
                                                      const std::unordered_set<std::string> *collective_nodes,
                                                      void *gpu_stream)
    {
        if (!capture)
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph capture is null, falling back to fast decode");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // TP>1 with collectives cannot be captured in a single-device graph
        if (collective_nodes && !collective_nodes->empty())
        {
            LOG_DEBUG("[DeviceGraphExecutor] Collective nodes present, skipping graph capture for TP>1");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // Step 1: Begin capture
        if (!capture->beginCapture())
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph beginCapture failed, falling back to fast decode");
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // Step 2: Execute all stages into the captured stream
        // Set GPU device once before the loop (same as executeFastDecode)
        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

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
                LOG_ERROR("[DeviceGraphExecutor] Stage failed during graph capture: " << name);
                exec_success = false;
                break;
            }
            graph.markCompleted(name);
        }

        // Step 3: End capture
        if (!exec_success || !capture->endCapture())
        {
            LOG_WARN("[DeviceGraphExecutor] GPU graph capture failed (exec_success=" << exec_success
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
            LOG_WARN("[DeviceGraphExecutor] GPU graph captured 0 nodes — kernels NOT on capture stream! Skipping graph replay");
            return true;
        }

        // Step 4: Instantiate or update + launch
        LOG_WARN("[DeviceGraphExecutor] GPU graph captured " << capture->nodeCount()
                                                             << " nodes, hasExecutable=" << capture->hasExecutable());
        if (capture->hasExecutable())
        {
            // Try in-place update
            GraphUpdateResult result = capture->tryUpdate();
            LOG_WARN("[DeviceGraphExecutor] tryUpdate result=" << static_cast<int>(result));
            if (result == GraphUpdateResult::Success)
            {
                // In-place update succeeded — launch the updated executable
                if (!capture->launch())
                {
                    LOG_ERROR("[DeviceGraphExecutor] GPU graph launch failed after update");
                    return false;
                }
                LOG_TRACE("[DeviceGraphExecutor] GPU graph updated and launched ("
                          << capture->nodeCount() << " nodes)");
                return true;
            }
            else if (result == GraphUpdateResult::NeedsReinstantiate)
            {
                // Topology changed — reinstantiate
                LOG_WARN("[DeviceGraphExecutor] NeedsReinstantiate — calling instantiate()");
                if (!capture->instantiate())
                {
                    LOG_WARN("[DeviceGraphExecutor] GPU graph reinstantiation failed");
                    return false;
                }
            }
            else
            {
                // Update failed
                LOG_WARN("[DeviceGraphExecutor] GPU graph update failed (result="
                         << static_cast<int>(result) << ")");
                return false;
            }
        }
        else
        {
            // First time — instantiate from captured graph
            LOG_WARN("[DeviceGraphExecutor] First instantiation attempt (" << capture->nodeCount() << " nodes)");
            if (!capture->instantiate())
            {
                LOG_WARN("[DeviceGraphExecutor] GPU graph instantiation failed");
                return false;
            }
            LOG_WARN("[DeviceGraphExecutor] GPU graph instantiated with " << capture->nodeCount()
                                                                          << " nodes (" << capture->backendName() << ")");
        }

        // Launch the (newly instantiated) executable
        if (!capture->launch())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph launch failed");
            return false;
        }

        return true;
    }

    bool DeviceGraphExecutor::executeDecodeWithCapturePolicy(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        GraphSegmentCache *segment_cache,
        void *gpu_stream,
        IWorkerGPUContext *gpu_ctx,
        const std::unordered_set<std::string> *collective_nodes,
        const DecodeCapturePolicy &policy,
        bool *used_segmented_capture)
    {
        if (used_segmented_capture)
        {
            *used_segmented_capture = false;
        }

        if (!policy.allow_fast_decode)
        {
            return execute(graph, ctx);
        }

        const bool segmented_ready =
            policy.allow_segmented_capture &&
            segment_cache &&
            gpu_stream &&
            gpu_ctx &&
            segment_cache->consecutive_failures < policy.max_segment_failures;

        if (segmented_ready)
        {
            bool success = executeWithSegmentedGraphCapture(
                graph,
                ctx,
                *segment_cache,
                gpu_stream,
                gpu_ctx,
                collective_nodes,
                policy.collectives_graph_capturable);

            if (success)
            {
                if (used_segmented_capture)
                {
                    *used_segmented_capture = true;
                }
                return true;
            }

            LOG_WARN("[DeviceGraphExecutor] Segmented replay failed under policy, falling back to fast decode");
            graph.reset();
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        return executeFastDecode(graph, ctx, collective_nodes);
    }

    bool DeviceGraphExecutor::executeWithSegmentedGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                               GraphSegmentCache &segment_cache,
                                                               void *gpu_stream,
                                                               IWorkerGPUContext *gpu_ctx,
                                                               const std::unordered_set<std::string> *collective_nodes,
                                                               bool collectives_graph_capturable)
    {
        if (!gpu_stream || !gpu_ctx)
        {
            LOG_WARN("[DeviceGraphExecutor] Segmented graph capture: missing stream or gpu_ctx, falling back");
            return executeFastDecode(graph, ctx);
        }

        const bool has_collective_nodes = (collective_nodes && !collective_nodes->empty());

        // Monotonic step counter + phase transition selection for segmented mode.
        const auto phase_transition = DeviceGraphCaptureController::beginStep(
            segment_cache.initialized,
            segment_cache.needs_capture,
            segment_cache.decode_step);
        const uint64_t current_step = phase_transition.decode_step;

        auto mark_stage_outputs_dirty = [&](ComputeNode &node, void *stream)
        {
            const auto &dump_info = node.stage->getDumpInfo();
            auto outputs = extractOutputBuffers(dump_info);
            markOutputsDirty(outputs, stream);
        };

        auto post_captured_segment_launch = [&](GraphSegment &seg, void *stream)
        {
            DeviceGraphCaptureController::postCapturedSegmentLaunch(
                graph,
                seg,
                current_step,
                stream,
                [&](ComputeNode &node, void *node_stream)
                {
                    mark_stage_outputs_dirty(node, node_stream);
                });
        };

        auto cohere_replay_stage = [&](ComputeNode &node) -> bool
        {
            const auto policy = node.stage->coherencePolicy();
            if (policy != CoherencePolicy::INPUT && policy != CoherencePolicy::FULL)
            {
                return true;
            }

            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            const auto &dump_info = node.stage->getDumpInfo();

            auto inputs = extractInputBuffers(dump_info);
            if (!cohereInputs(inputs, target_device))
            {
                LOG_ERROR("[DeviceGraphExecutor] Failed to cohere replay inputs for stage: " << node.name);
                return false;
            }

            auto weights = extractWeightBuffers(dump_info);
            if (!cohereInputs(weights, target_device))
            {
                LOG_ERROR("[DeviceGraphExecutor] Failed to cohere replay weights for stage: " << node.name);
                return false;
            }

            if (policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL)
            {
                auto outputs = extractOutputBuffers(dump_info);
                if (!cohereOutputs(outputs, target_device))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Failed to cohere replay outputs for stage: " << node.name);
                    return false;
                }
            }

            return true;
        };

        auto cohere_segment_inputs = [&](const GraphSegment &seg) -> bool
        {
            return DeviceGraphCaptureController::cohereReplaySegmentInputs(
                graph,
                seg,
                [&](ComputeNode &node)
                {
                    return cohere_replay_stage(node);
                });
        };

        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

        DeviceGraphCaptureController::ReplayHooks replay_hooks{
            [&](const GraphSegment &segment)
            {
                return cohere_segment_inputs(segment);
            },
            [&](ComputeNode &node)
            {
                return executeNode(node, ctx);
            },
            [&](GraphSegment &segment, void *stream)
            {
                post_captured_segment_launch(segment, stream);
            }};

        // Capture-phase hooks: same as replay hooks except post_launch skips
        // onGraphReplayed() callbacks. During capture, execute() already ran
        // host-side bookkeeping; calling onGraphReplayed() would double-advance
        // KV cache head positions and corrupt subsequent decode steps.
        DeviceGraphCaptureController::ReplayHooks capture_hooks{
            replay_hooks.cohere_inputs,
            replay_hooks.execute_node,
            [&](GraphSegment &segment, void *stream)
            {
                DeviceGraphCaptureController::postCapturedSegmentLaunch(
                    graph,
                    segment,
                    current_step,
                    stream,
                    [&](ComputeNode &node, void *node_stream)
                    {
                        mark_stage_outputs_dirty(node, node_stream);
                    },
                    /*skip_replay_callbacks=*/true);
            }};

        // ===== Phase 1: Warmup (first call) — build segments, execute normally =====
        // We do NOT capture on the first call. Some kernels lazily initialize workspace
        // buffers (hipMalloc), which isn't compatible with stream capture.
        // First call builds the segment list and runs via executeFastDecode.
        //
        // CRITICAL: Run warmup on the CAPTURE stream (not default stream). CK and
        // ROCm kernel dispatch may cache per-stream state (dispatch tables, workspace
        // allocations). If warmup runs on the default stream, the capture stream sees
        // "fresh" kernel contexts that trigger capture-unsafe lazy initialization,
        // causing intermittent "operation failed due to a previous error during capture".
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Warmup)
        {
            DeviceGraphCaptureController::executeWarmupPhase(
                graph,
                segment_cache,
                collective_nodes,
                has_collective_nodes,
                collectives_graph_capturable);

            // Create the capture stream early so warmup runs on it.
            if (segment_cache.ensureCaptureStream(gpu_ctx))
            {
                void *warmup_stream = segment_cache.capture_stream;
                // Point all stages at the capture stream for warmup execution.
                for (const auto &name : graph.getExecutionOrder())
                {
                    auto *node = graph.getNode(name);
                    if (node && node->stage)
                    {
                        node->stage->setGPUStream(warmup_stream);
                    }
                }
            }

            // Warmup executes all stages normally (no capture) to ensure
            // lazy kernel initialization and workspace allocation complete
            // on the capture stream.
            return executeFastDecode(graph, ctx, collective_nodes);
        }

        // ===== Phase 2: Capture (second call) — record capturable segments =====
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Capture)
        {
            const auto capture_result = DeviceGraphCaptureController::executeCapturePhase(
                graph,
                segment_cache,
                ctx,
                gpu_ctx,
                has_collective_nodes,
                current_step,
                capture_hooks);

            if (capture_result.reset_cache)
            {
                segment_cache.reset();
            }

            if (!capture_result.success)
            {
                if (capture_result.fallback_to_fast_decode)
                {
                    // The partial capture marked some nodes completed —
                    // reset so fast decode re-executes all stages.
                    graph.reset();
                    return executeFastDecode(graph, ctx, collective_nodes);
                }
                return false;
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

        const auto replay_phase_result = DeviceGraphCaptureController::executeReplayPhase(
            graph,
            segment_cache,
            ctx,
            gpu_ctx,
            has_collective_nodes,
            current_step,
            replay_hooks);

        if (!replay_phase_result.success)
        {
            if (replay_phase_result.launch_failure_fallback)
            {
                segment_cache.consecutive_failures++;
                if (segment_cache.consecutive_failures >= GraphSegmentCache::kMaxFailures)
                {
                    LOG_WARN("[DeviceGraphExecutor] Too many segmented graph failures, disabling");
                    segment_cache.reset();
                }
                graph.reset();
                return executeFastDecode(graph, ctx, collective_nodes);
            }
            return false;
        }

        segment_cache.consecutive_failures = 0;
        return true;
    }

    bool DeviceGraphExecutor::executeParallel(ComputeGraph &graph, IDeviceContext *ctx)
    {
        auto total_start = std::chrono::high_resolution_clock::now();

        while (!graph.allCompleted())
        {
            auto ready = graph.getReadyNodes();

            if (ready.empty() && !graph.allCompleted())
            {
                LOG_ERROR("[DeviceGraphExecutor] Deadlock detected in graph");
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
                    LOG_ERROR("[DeviceGraphExecutor] Stage failed: " << name);
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

    bool DeviceGraphExecutor::executeMultiDevice(
        ComputeGraph &graph,
        const std::unordered_map<DeviceId, IDeviceContext *> &contexts)
    {

        if (contexts.empty())
        {
            LOG_ERROR("[DeviceGraphExecutor] No device contexts provided");
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

        // Mark the last stage as needing event-based dirty marking
        // (its outputs will be read by CPU for sampling/logits)
        if (!order.empty())
        {
            auto *last_node = graph.getNode(order.back());
            if (last_node)
                last_node->is_final_output = true;
        }

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
                LOG_ERROR("[DeviceGraphExecutor] Stage failed: " << name << " on device " << node->device.to_string());
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

#ifdef HAVE_ROCM
    static void logWatchedPointerProducer(const std::string &stage_name, const StageDumpInfo &dump_info)
    {
        const auto &validation = debugEnv().validation;
        if (!validation.trace_local_tp_pointer)
        {
            return;
        }

        const uintptr_t watch = static_cast<uintptr_t>(validation.trace_local_tp_pointer_address);
        for (const auto &output : dump_info.outputs)
        {
            if (!output.tensor)
            {
                continue;
            }

            auto *tb = dynamic_cast<TensorBase *>(output.tensor);
            if (!tb)
            {
                continue;
            }

            void *gpu_ptr = tb->gpu_data_ptr();
            if (!gpu_ptr)
            {
                continue;
            }

            ROCmPointerOwnerInfo owner;
            if (!ROCmBackend::queryPointerOwner(gpu_ptr, owner) || !owner.active)
            {
                continue;
            }

            const uintptr_t begin = reinterpret_cast<uintptr_t>(owner.base_ptr);
            const uintptr_t end = begin + owner.size_bytes;
            if (watch < begin || watch >= end)
            {
                continue;
            }

            const size_t offset = static_cast<size_t>(watch - begin);
            LOG_WARN("[LOCALTP_PTR_PRODUCER]"
                     << " stage=" << stage_name
                     << " output=" << (output.name ? output.name : "(unnamed)")
                     << " watch=" << reinterpret_cast<const void *>(watch)
                     << " output_ptr=" << gpu_ptr
                     << " owner_base=" << owner.base_ptr
                     << " owner_bytes=" << owner.size_bytes
                     << " owner_device=" << owner.device_id
                     << " owner_seq=" << owner.sequence
                     << " owner_thread=" << owner.thread_hash
                     << " offset=" << offset
                     << " tensor=" << static_cast<void *>(tb)
                     << " tensor_name=" << (tb->debugName().empty() ? "(unnamed)" : tb->debugName()));
        }
    }
#endif

    bool DeviceGraphExecutor::executeNode(ComputeNode &node, IDeviceContext *ctx)
    {
        if (!node.stage)
        {
            LOG_ERROR("[DeviceGraphExecutor] Node '" << node.name << "' has no stage");
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
                LOG_DEBUG("[DeviceGraphExecutor] Intercepting ALLREDUCE stage '" << node.name << "' via CollectiveContext");
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
                    LOG_DEBUG("[DeviceGraphExecutor] Skipping strided ALLGATHER intercept in segmented collective mode for '" << node.name << "'");
                }
                else
                {
                    LOG_DEBUG("[DeviceGraphExecutor] Attempting strided ALLGATHER intercept for '" << node.name << "'");
                    if (executeCollectiveStridedAllgather(node, ctx))
                    {
                        return true;
                    }
                    // Fall through to normal execution if strided path not available
                    LOG_DEBUG("[DeviceGraphExecutor] Strided ALLGATHER not available, using stage execution");
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

            LOG_DEBUG("[DeviceGraphExecutor] Stage '" << node.name << "' coherencePolicy=" << toString(policy)
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
                    LOG_ERROR("[DeviceGraphExecutor] Failed to cohere inputs for stage '" << node.name << "'");
                    return false;
                }
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    input_cohere_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }

                // Cohere weights (needed for GPU execution - biases, etc.)
                // OPTIMIZATION: Skip if weights were already confirmed on device
                // (weights are immutable after loading, so first-pass coherence is sufficient)
                if (!node.weights_cohered)
                {
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
                        LOG_ERROR("[DeviceGraphExecutor] Failed to cohere weights for stage '" << node.name << "'");
                        return false;
                    }
                    if (profiling)
                    {
                        phase_end = std::chrono::high_resolution_clock::now();
                        weight_cohere_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    }
                    node.weights_cohered = true;
                }
            }

            // For GPU targets, outputs also need GPU buffers allocated before kernel runs
            // OPTIMIZATION: Skip if outputs were already allocated on device
            // (GPU output buffers persist across iterations, never need re-allocation)
            if ((policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL) && !node.outputs_allocated)
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
                    LOG_ERROR("[DeviceGraphExecutor] Failed to allocate output buffers for stage '" << node.name << "'");
                    return false;
                }
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    output_alloc_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }

                // Cache output buffers for mark_dirty (avoid re-extracting every iteration)
                node.cached_output_buffers = std::move(outputs);
                node.outputs_allocated = true;
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

        // =========================================================================
        // GPU Pointer Device Validation (diagnostic for multi-GPU memory faults)
        // Validates all GPU pointers belong to the expected device before execution.
        // Gated by LLAMINAR_VALIDATE_GPU_PTRS=1 to avoid hipPointerGetAttributes overhead.
        // =========================================================================
#ifdef HAVE_ROCM
        if (debugEnv().validation.validate_gpu_ptrs)
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            if (target_device.is_rocm())
            {
                int expected_ordinal = target_device.rocm_ordinal();
                bool ptr_validation_failed = false;
                auto validatePtr = [&](const char *label, const char *tensor_name, void *gpu_ptr)
                {
                    if (!gpu_ptr)
                        return;
                    hipPointerAttribute_t attr = {};
                    hipError_t err = hipPointerGetAttributes(&attr, gpu_ptr);
                    if (err == hipSuccess && attr.device != expected_ordinal)
                    {
                        ptr_validation_failed = true;
                        ROCmPointerOwnerInfo owner;
                        const bool has_owner = ROCmBackend::queryPointerOwner(gpu_ptr, owner);
                        if (has_owner)
                        {
                            LOG_ERROR("[GPU_PTR_VIOLATION] Stage='" << node.name
                                                                    << "' tensor=" << tensor_name << " (" << label << ")"
                                                                    << " gpu_ptr=" << gpu_ptr
                                                                    << " attr.device=" << attr.device
                                                                    << " expected=" << expected_ordinal
                                                                    << " target=" << target_device.to_string()
                                                                    << " owner.base=" << owner.base_ptr
                                                                    << " owner.dev=" << owner.device_id
                                                                    << " owner.bytes=" << owner.size_bytes
                                                                    << " owner.seq=" << owner.sequence
                                                                    << " owner.thread=" << owner.thread_hash
                                                                    << " owner.active=" << owner.active);
                        }
                        else
                        {
                            LOG_ERROR("[GPU_PTR_VIOLATION] Stage='" << node.name
                                                                    << "' tensor=" << tensor_name << " (" << label << ")"
                                                                    << " gpu_ptr=" << gpu_ptr
                                                                    << " attr.device=" << attr.device
                                                                    << " expected=" << expected_ordinal
                                                                    << " target=" << target_device.to_string()
                                                                    << " owner=unknown");
                        }
                        ROCmBackend::dumpRecentPointerEvents(48);
                    }
                    else if (err != hipSuccess)
                    {
                        ptr_validation_failed = true;
                        ROCmPointerOwnerInfo owner;
                        const bool has_owner = ROCmBackend::queryPointerOwner(gpu_ptr, owner);
                        LOG_ERROR("[GPU_PTR_ATTR_ERROR] Stage='" << node.name
                                                                 << "' tensor=" << tensor_name << " (" << label << ")"
                                                                 << " gpu_ptr=" << gpu_ptr
                                                                 << " hipError=" << hipGetErrorString(err)
                                                                 << (has_owner ? (std::string(" owner.dev=") + std::to_string(owner.device_id) +
                                                                                  " owner.bytes=" + std::to_string(owner.size_bytes) +
                                                                                  " owner.seq=" + std::to_string(owner.sequence))
                                                                               : std::string(" owner=unknown")));
                    }
                };

                // Validate inputs from dump info
                for (const auto &input : cached_dump_info.inputs)
                {
                    if (input.tensor)
                    {
                        if (auto *tb = dynamic_cast<TensorBase *>(const_cast<ITensor *>(input.tensor)))
                        {
                            validatePtr("input", input.name ? input.name : "(unnamed)", tb->gpu_data_ptr());
                        }
                    }
                }
                // Validate outputs from dump info
                for (const auto &output : cached_dump_info.outputs)
                {
                    if (output.tensor)
                    {
                        if (auto *tb = dynamic_cast<TensorBase *>(output.tensor))
                        {
                            validatePtr("output", output.name ? output.name : "(unnamed)", tb->gpu_data_ptr());
                        }
                    }
                }
                // Validate weights from dump info
                for (const auto &weight : cached_dump_info.weights)
                {
                    if (weight.tensor)
                    {
                        if (auto *tb = dynamic_cast<TensorBase *>(const_cast<ITensor *>(weight.tensor)))
                        {
                            validatePtr("weight", weight.name ? weight.name : "(unnamed)", tb->gpu_data_ptr());
                        }
                    }
                }

                if (ptr_validation_failed)
                {
                    LOG_ERROR("[GPU_PTR_VIOLATION_ABORT] Aborting stage execute before kernel launch: stage='"
                              << node.name << "' target=" << target_device.to_string()
                              << " expected_rocm_ordinal=" << expected_ordinal);
                    return false;
                }
            }
        }
#endif

        if (profiling)
            phase_start = std::chrono::high_resolution_clock::now();

        ensureStageGPUStreamBound(node, ctx);

        bool success = node.stage->execute(ctx);

#ifdef HAVE_ROCM
        if (success && debugEnv().validation.sync_each_stage)
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            if (target_device.is_rocm())
            {
                HipDeviceGuard::forceSetDevice(target_device.rocm_ordinal());
                hipError_t sync_err = hipDeviceSynchronize();
                if (sync_err != hipSuccess)
                {
                    LOG_ERROR("[SYNC_EACH_STAGE] stage='" << node.name
                                                          << "' device=" << target_device.to_string()
                                                          << " hipDeviceSynchronize failed: "
                                                          << hipGetErrorString(sync_err));
                    ROCmBackend::dumpRecentPointerEvents(96);
                    success = false;
                }
            }
        }
#endif

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
                // OPTIMIZATION: Use cached output buffers instead of re-extracting
                // The cached_output_buffers were populated during the first-pass output allocation
                if (node.cached_output_buffers.empty())
                {
                    // First time or cache not populated — extract and cache
                    if (profiling)
                        phase_start = std::chrono::high_resolution_clock::now();
                    node.cached_output_buffers = extractOutputBuffers(cached_dump_info);
                    if (profiling)
                    {
                        phase_end = std::chrono::high_resolution_clock::now();
                        extract_buffers_ms += std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                        phase_start = phase_end;
                    }
                }
                else if (profiling)
                {
                    phase_start = std::chrono::high_resolution_clock::now();
                }

                // OPTIMIZATION: Use flags-only dirty marking for intermediate stages.
                // Only the final stage (producing logits for CPU readback) needs event
                // recording. Intermediate GPU→GPU stages skip hipEventRecord overhead.
                if (node.is_final_output)
                {
                    markOutputsDirty(node.cached_output_buffers, node.stage->gpuStream());
                }
                else
                {
                    markOutputsDirtyFlagsOnly(node.cached_output_buffers);
                }
                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    mark_dirty_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }

#ifdef HAVE_ROCM
                logWatchedPointerProducer(node.name, cached_dump_info);
#endif

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
        LOG_DEBUG("[DeviceGraphExecutor::executeNode] success=" << success << " callback=" << (config_.snapshot_callback ? "set" : "null") << " node=" << node.name);
        if (success && config_.snapshot_callback)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            // IMPORTANT: Sync outputs from GPU before callback reads them
            cached_dump_info.ensureOutputsOnHost();
            LOG_DEBUG("[DeviceGraphExecutor::executeNode] Invoking callback for " << node.name);
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
            LOG_TRACE("[DeviceGraphExecutor::PHASES] " << node.name
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

            LOG_DEBUG("[DeviceGraphExecutor] Stage '" << node.name << "' took " << total_ms << " ms (execute=" << execute_ms << "ms, overhead=" << total_overhead_ms << "ms)");
        }

        return success;
    }

    // =============================================================================
    // Buffer Validation (Debug/Integration Builds Only)
    // =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE

    void DeviceGraphExecutor::verifyStageEntry(const ComputeNode &node, int layer_idx)
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

    void DeviceGraphExecutor::verifyStageExit(const ComputeNode &node, int layer_idx)
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

    bool DeviceGraphExecutor::validateStageOutputs(const ComputeNode &node)
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
                        LOG_DEBUG("[DeviceGraphExecutor] Skipping GPU validation for BAR-backed tensor '"
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
                                            LOG_WARN("[DeviceGraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                                                     << "' appears to be all zeros (GPU validation)");
                                            if (validation.fail_on_zero)
                                            {
                                                LOG_ERROR("[DeviceGraphExecutor] Buffer validation failed: zero tensor detected");
                                                all_valid = false;
                                            }
                                        }

                                        if (result.has_nan || result.has_inf)
                                        {
                                            LOG_WARN("[DeviceGraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                                                     << "' contains " << result.nan_count << " NaN, "
                                                                                     << result.inf_count << " Inf values (GPU validation)");
                                            if (validation.fail_on_nan)
                                            {
                                                LOG_ERROR("[DeviceGraphExecutor] Buffer validation failed: NaN/Inf detected");
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
                    LOG_WARN("[DeviceGraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                             << "' appears to be all zeros (likely uninitialized)");

                    if (validation.fail_on_zero)
                    {
                        LOG_ERROR("[DeviceGraphExecutor] Buffer validation failed: zero tensor detected");
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
                    LOG_WARN("[DeviceGraphExecutor] Stage '" << node.name << "' output '" << output.name
                                                             << "' contains NaN or Inf values");

                    if (validation.fail_on_nan)
                    {
                        LOG_ERROR("[DeviceGraphExecutor] Buffer validation failed: NaN/Inf detected");
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

    bool DeviceGraphExecutor::executeWithBufferManagement(ComputeGraph &graph, IDeviceContext *ctx)
    {
        if (!buffer_manager_)
        {
            LOG_ERROR("[DeviceGraphExecutor] executeWithBufferManagement called without buffer manager set");
            return false;
        }

        LOG_DEBUG("[DeviceGraphExecutor] Allocating buffers for graph...");

        // Allocate all buffers based on stage requirements
        if (!buffer_manager_->allocateForGraph(graph))
        {
            LOG_ERROR("[DeviceGraphExecutor] Failed to allocate buffers for graph");
            return false;
        }

        LOG_DEBUG("[DeviceGraphExecutor] Allocated " << buffer_manager_->bufferCount()
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

    bool DeviceGraphExecutor::executeCollectiveAllreduce(ComputeNode &node, IDeviceContext *ctx)
    {
        (void)ctx; // Device context not needed - CollectiveContext handles device

        auto *stage = dynamic_cast<AllreduceStage *>(node.stage.get());
        if (!stage)
        {
            LOG_ERROR("[DeviceGraphExecutor] Failed to cast stage '" << node.name << "' to AllreduceStage");
            return false;
        }

        // Get buffer info from stage's dump info
        auto dump_info = stage->getDumpInfo();
        if (dump_info.inputs.empty() || !dump_info.inputs[0].tensor)
        {
            LOG_ERROR("[DeviceGraphExecutor] AllreduceStage '" << node.name << "' has no input buffer");
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
            LOG_DEBUG("[DeviceGraphExecutor] Executing allreduce in domain: " << domain->name);
            success = collective_ctx_->executeAllreduceInDomain(
                buffer,
                count,
                tensor_device,
                CollectiveOp::ALLREDUCE_SUM,
                domain);
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Executing allreduce via legacy (no domain) path");
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
            LOG_DEBUG("[DeviceGraphExecutor] ALLREDUCE '" << node.name << "' via CollectiveContext took " << ms << " ms");
        }

        if (!success)
        {
            LOG_ERROR("[DeviceGraphExecutor] CollectiveContext::executeAllreduce failed for '" << node.name << "'");
        }

        return success;
    }

    bool DeviceGraphExecutor::executeCollectiveAllgather(ComputeNode &node, IDeviceContext *ctx)
    {
        (void)ctx; // Device context not needed - CollectiveContext handles device

        auto *stage = dynamic_cast<AllGatherStage *>(node.stage.get());
        if (!stage)
        {
            LOG_ERROR("[DeviceGraphExecutor] Failed to cast stage '" << node.name << "' to AllGatherStage");
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
            LOG_ERROR("[DeviceGraphExecutor] AllGatherStage '" << node.name << "' missing input or output buffer");
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
            LOG_DEBUG("[DeviceGraphExecutor] Executing allgather in domain: " << domain->name);
            success = collective_ctx_->executeAllgatherInDomain(
                local_input,
                full_output,
                actual_seq_len,
                tensor_device,
                domain);
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Executing allgather via legacy (no domain) path");
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
            LOG_DEBUG("[DeviceGraphExecutor] ALLGATHER '" << node.name << "' via CollectiveContext took " << ms << " ms");
        }

        if (!success)
        {
            LOG_ERROR("[DeviceGraphExecutor] CollectiveContext::executeAllgather failed for '" << node.name << "'");
        }

        return success;
    }

    bool DeviceGraphExecutor::executeCollectiveStridedAllgather(ComputeNode &node, IDeviceContext *ctx)
    {
        (void)ctx; // Device context not needed - CollectiveContext handles device

        auto *stage = dynamic_cast<AllGatherStage *>(node.stage.get());
        if (!stage)
        {
            LOG_ERROR("[DeviceGraphExecutor] Failed to cast stage '" << node.name << "' to AllGatherStage");
            return false;
        }

        // Get parameters directly from stage
        const auto &params = stage->getParams();

        ITensor *local_input = params.local_input;
        ITensor *full_output = params.full_output;

        if (!local_input || !full_output)
        {
            LOG_DEBUG("[DeviceGraphExecutor] AllGatherStage '" << node.name << "' missing input or output buffer");
            return false;
        }

        // Use actual_seq_len from params, fallback to buffer rows
        size_t actual_seq_len = params.actual_seq_len > 0 ? params.actual_seq_len : local_input->rows();

        // Determine device where tensors reside
        DeviceId tensor_device = node.device.is_valid() ? node.device : DeviceId::cpu();

        // Strided allgather only works on CUDA
        if (tensor_device.type != DeviceType::CUDA)
        {
            LOG_DEBUG("[DeviceGraphExecutor] Strided allgather requires CUDA device, falling back");
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
            LOG_DEBUG("[DeviceGraphExecutor] Strided ALLGATHER '" << node.name << "' via NCCL took " << ms << " ms");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Strided allgather not available for '" << node.name << "'");
        }

        return success;
    }

    // =============================================================================
    // Workspace Management
    // =============================================================================

    float *DeviceGraphExecutor::getTemporaryBuffer(size_t elements)
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
