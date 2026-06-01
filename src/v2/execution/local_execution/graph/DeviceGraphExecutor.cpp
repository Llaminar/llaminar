/**
 * @file DeviceGraphExecutor.cpp
 * @brief Compute graph execution engine implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "DeviceGraphExecutor.h"
#include "StageVerifier.h"
#include "DeviceGraphCaptureController.h"
#include "GraphCaptureGuard.h"
#include "../../debug/StageDumper.h"
#include "../../debug/AsyncStageDumper.h"
#include "../coherence/StageCoherence.h"
#include "../collective/CollectiveContext.h"
#include "../../compute_stages/stages/AllreduceStage.h"
#include "../../compute_stages/stages/AllGatherStage.h"
#include "../../../config/TPDomain.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <optional>
#include <print>
#include "fort.hpp"
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <cstdint>

namespace llaminar2
{
    namespace
    {
        IWorkerGPUContext *tryGetWorkerContext(const DeviceId &device)
        {
            if (!device.is_gpu())
            {
                return nullptr;
            }

            try
            {
                return &GPUDeviceContextPool::instance().getContext(device);
            }
            catch (const std::exception &e)
            {
                LOG_DEBUG("[DeviceGraphExecutor] Failed to resolve worker GPU context for "
                          << device.to_string() << ": " << e.what());
                return nullptr;
            }
        }

        void *resolveWorkerDefaultStream(const DeviceId &device)
        {
            if (auto *gpu_ctx = tryGetWorkerContext(device))
            {
                return gpu_ctx->defaultStream();
            }

            return nullptr;
        }

        bool validateStagePointerSet(
            IWorkerGPUContext *gpu_ctx,
            const std::string &stage_name,
            const char *label,
            int expected_ordinal,
            ITensor *tensor,
            const char *tensor_name,
            bool dump_pointer_events)
        {
            if (!gpu_ctx || !tensor)
            {
                return true;
            }

            auto *tb = dynamic_cast<TensorBase *>(tensor);
            if (!tb)
            {
                return true;
            }

            void *gpu_ptr = tb->gpu_data_ptr();
            if (!gpu_ptr)
            {
                return true;
            }

            const auto validation = gpu_ctx->validatePointerDevice(gpu_ptr, expected_ordinal);
            if (validation.valid)
            {
                return true;
            }

            LOG_ERROR("[GPU_PTR_VIOLATION] Stage='" << stage_name
                                                    << "' tensor=" << (tensor_name ? tensor_name : "(unnamed)")
                                                    << " (" << label << ")"
                                                    << " gpu_ptr=" << gpu_ptr
                                                    << " actual=" << validation.actual_device
                                                    << " expected=" << expected_ordinal
                                                    << " " << validation.details);

            if (dump_pointer_events)
            {
                gpu_ctx->dumpRecentPointerEvents(48);
            }

            return false;
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
    // GraphSegmentCache & GPU graph capture implementations moved to
    // DeviceGraphExecutor_GraphCapture.cpp
    // =========================================================================

    // Forward declarations for static helpers used by runStage()
    static void printStageOutputs(const std::string &stage_name, const StageDumpInfo &dump_info);
    static void logWatchedPointerProducer(
        const std::string &stage_name,
        const StageDumpInfo &dump_info,
        const IWorkerGPUContext *gpu_ctx);

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

    thread_local ExecutionPhase GraphExecutorStats::current_phase_ = ExecutionPhase::COMBINED;

    void GraphExecutorStats::printPhaseTable(const std::string &title, const PhaseStats &phase, size_t tokens) const
    {
        if (phase.total_stages_executed == 0)
            return;

        auto fmt = [](double val, int prec) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        auto per_tok = [&](double val) -> std::string
        {
            if (tokens == 0)
                return "-";
            return fmt(val / tokens, 3);
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title row
        {
            std::ostringstream oss;
            oss << title << " (" << tokens << " tokens)";
            table << oss.str() << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
        }

        // Header
        table << fort::header << "STAGE TYPE" << "CALLS" << "TOTAL (ms)" << "PER-TOKEN (ms)" << "%" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);
        table.column(4).set_cell_text_align(fort::text_align::right);

        // Sort by time descending
        std::vector<std::pair<std::string, double>> rows(
            phase.stage_type_execute_ms.begin(), phase.stage_type_execute_ms.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto &a, const auto &b)
                  { return a.second > b.second; });

        for (const auto &[stage_type, ms] : rows)
        {
            size_t count = 0;
            auto it = phase.stage_type_counts.find(stage_type);
            if (it != phase.stage_type_counts.end())
                count = it->second;
            double share = phase.total_execute_ms > 0 ? (ms / phase.total_execute_ms) * 100.0 : 0;
            table << stage_type << std::to_string(count) << fmt(ms, 2) << per_tok(ms) << (fmt(share, 1) + "%") << fort::endr;
        }

        // Total + overhead
        double phase_overhead = phase.overhead.total();
        double phase_all = phase.total_execute_ms + phase_overhead;
        table << fort::separator;
        table << "TOTAL KERNEL" << "" << fmt(phase.total_execute_ms, 2) << per_tok(phase.total_execute_ms) << "" << fort::endr;
        if (phase_overhead > 0.01)
        {
            table << "TOTAL OVERHEAD" << "" << fmt(phase_overhead, 2) << per_tok(phase_overhead) << "" << fort::endr;
        }

        // Throughput
        if (tokens > 0 && phase_all > 0)
        {
            double toks_per_sec = (tokens / phase_all) * 1000.0;
            std::ostringstream oss;
            oss << fmt(toks_per_sec, 2) << " tok/s  |  " << fmt(phase_all / tokens, 3) << " ms/token";
            if (phase.total_collective_calls > 0)
                oss << "  |  collective: " << fmt(phase.total_collective_ms, 2) << " ms (" << phase.total_collective_calls << " calls)";
            table << fort::separator;
            table << oss.str() << "" << "" << "" << "" << fort::endr;
            table[table.row_count() - 1][0].set_cell_span(5);
        }

        std::print("\n{}", table.to_string());
    }

    void GraphExecutorStats::printProfilingSummary(size_t prefill_tokens, size_t decode_tokens) const
    {
        // Print per-phase tables first (the useful ones)
        printPhaseTable("EXECUTOR STAGE PROFILING — PREFILL", prefill, prefill_tokens);
        printPhaseTable("EXECUTOR STAGE PROFILING — DECODE", decode, decode_tokens);

        // Calculate totals for overhead summary
        double total_overhead = overhead.total();
        double total_all = total_execute_ms + total_overhead;
        size_t total_tokens = prefill_tokens + decode_tokens;

        auto fmt = [](double val, int prec) -> std::string
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec) << val;
            return oss.str();
        };

        auto pct = [&](double val) -> std::string
        {
            double p = total_all > 0 ? (val / total_all) * 100.0 : 0;
            return fmt(p, 1) + "%";
        };

        auto per_tok = [&](double val) -> std::string
        {
            if (total_tokens == 0)
                return "-";
            return fmt(val / total_tokens, 3);
        };

        // Combined overhead table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        table << "EXECUTOR OVERHEAD SUMMARY (COMBINED)" << "" << "" << "" << fort::endr;
        table[0][0].set_cell_span(4);
        table[0][0].set_cell_text_align(fort::text_align::center);

        {
            std::ostringstream oss;
            oss << "Total stages: " << total_stages_executed
                << "  |  Prefill: " << prefill.total_stages_executed
                << " (" << prefill_tokens << " tok)"
                << "  |  Decode: " << decode.total_stages_executed
                << " (" << decode_tokens << " tok)";
            table << oss.str() << "" << "" << "" << fort::endr;
            table[1][0].set_cell_span(4);
        }

        table << fort::header << "CATEGORY" << "TOTAL (ms)" << "PER-TOKEN (ms)" << "%" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);

        table << "Kernel Execution" << fmt(total_execute_ms, 2) << per_tok(total_execute_ms) << pct(total_execute_ms) << fort::endr;
        if (total_collective_calls > 0)
        {
            double compute_ms = total_execute_ms - total_collective_ms;
            table << "  Compute (kernels)" << fmt(compute_ms, 2) << per_tok(compute_ms) << pct(compute_ms) << fort::endr;
            std::ostringstream label;
            label << "  Collective (" << total_collective_calls << " calls)";
            table << label.str() << fmt(total_collective_ms, 2) << per_tok(total_collective_ms) << pct(total_collective_ms) << fort::endr;
        }

        // Coherence overhead
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

        // Efficiency
        table << fort::separator;
        double efficiency = total_all > 0 ? (total_execute_ms / total_all) * 100.0 : 0;
        double overhead_per_token = total_tokens > 0 ? total_overhead / total_tokens : 0;
        {
            std::ostringstream oss;
            oss << "Kernel Efficiency: " << fmt(efficiency, 1) << "%  (higher = less overhead)";
            if (total_tokens > 0)
                oss << "  |  Overhead per token: " << fmt(overhead_per_token, 3) << " ms";
            table << oss.str() << "" << "" << "" << fort::endr;
            table[table.row_count() - 1][0].set_cell_span(4);
        }

        if (total_collective_calls > 0)
        {
            double compute_ms = total_execute_ms - total_collective_ms;
            double compute_efficiency = total_all > 0 ? (compute_ms / total_all) * 100.0 : 0;
            std::ostringstream oss;
            oss << "Compute Efficiency: " << fmt(compute_efficiency, 1)
                << "%  (excluding " << fmt(total_collective_ms, 1) << " ms collective wait)";
            table << oss.str() << "" << "" << "" << fort::endr;
            table[table.row_count() - 1][0].set_cell_span(4);
        }

        std::print("\n{}", table.to_string());
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
            // PARALLEL runs sequentially (true parallel requires more infrastructure)
            return executeSequential(graph, ctx);
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
        return runStages(graph, ctx, StageRunPolicy::full());
    }

    bool DeviceGraphExecutor::executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                                                const std::unordered_set<std::string> *collective_nodes)
    {
        return runStages(graph, ctx, StageRunPolicy::fastDecode(), collective_nodes);
    }

    // executeWithGraphCapture, executeDecodeWithCapturePolicy,
    // executeWithSegmentedGraphCapture → DeviceGraphExecutor_GraphCapture.cpp

    // =========================================================================
    // Unified Stage Runner: runStages() + runStage()
    //
    // ALL execution paths (sequential, fast-decode, graph-capture manual segments)
    // funnel through these two methods. The StageRunPolicy controls which
    // phases are active, eliminating the class of bugs caused by divergent
    // code paths where one path does coherence/validation and another skips it.
    // =========================================================================

    bool DeviceGraphExecutor::cancellationRequested(const std::string &node_name) const
    {
        if (!config_.cancellation_requested || !config_.cancellation_requested())
            return false;

        LOG_WARN("[DeviceGraphExecutor] Execution canceled before stage: " << node_name);
        return true;
    }

    void DeviceGraphExecutor::notifyStageFailure(const std::string &node_name, const std::string &reason) const
    {
        if (config_.stage_failure_callback)
            config_.stage_failure_callback(node_name, reason);
    }

    bool DeviceGraphExecutor::runStages(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        const StageRunPolicy &policy,
        const std::unordered_set<std::string> *collective_nodes)
    {
        // Set GPU device once for the entire pass
        DeviceGraphCaptureController::prepareDeviceForSegmentedCapture(ctx);

        // =====================================================================
        // Build fast schedule (pre-computed flat array of {node*, is_collective})
        // Built once, reused across decode iterations. Eliminates string hash
        // lookups + markCompleted calls from the hot path.
        // =====================================================================
        if (!graph.hasFastSchedule())
        {
            graph.buildFastSchedule(collective_nodes);
        }
        const auto &schedule = graph.fastSchedule();
        if (schedule.empty())
            return true;

        // Ensure last stage is marked for event-based dirty marking
        schedule.back().node->is_final_output = true;

        // =====================================================================
        // GPU Stage Timing: event-based per-stage profiling
        // Gated by policy AND env var. ~1μs CPU overhead per event record.
        // =====================================================================
        const bool timeline_active = policy.timeline && debugEnv().gpu_stage_timing && ctx->isGPU() && !isGraphCaptureActive();
        IWorkerGPUContext *timeline_gpu_ctx = nullptr;
        void *timeline_stream = nullptr;
        if (timeline_active)
        {
            timeline_gpu_ctx = tryGetWorkerContext(ctx->deviceId());
            if (timeline_gpu_ctx)
            {
                timeline_stream = timeline_gpu_ctx->defaultStream();
                stage_timeline_.ensureCapacity(timeline_gpu_ctx, schedule.size());

                // Pre-populate stage metadata once — names/types never change
                if (!stage_timeline_info_populated_)
                {
                    for (size_t i = 0; i < schedule.size(); ++i)
                    {
                        auto *node = schedule[i].node;
                        if (node && node->stage)
                            stage_timeline_.setStageInfo(i, node->name, node->stage->type());
                    }
                    stage_timeline_info_populated_ = true;
                }
            }
        }

        auto total_start = std::chrono::high_resolution_clock::now();

        // =====================================================================
        // Main execution loop — every path goes through here
        // =====================================================================
        for (size_t i = 0; i < schedule.size(); ++i)
        {
            auto *node = schedule[i].node;
            const bool is_coll = schedule[i].is_collective;

            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphExecutor] Invalid node at schedule index " << i);
                notifyStageFailure("(invalid)", "invalid node in execution schedule");
                return false;
            }

            if (cancellationRequested(node->name))
                return false;

            ensureStageGPUStreamBound(*node, ctx);

            if (timeline_active && timeline_gpu_ctx)
                stage_timeline_.recordStart(i, timeline_gpu_ctx, timeline_stream);

            if (debugEnv().vram_trace && ctx->isGPU())
            {
                const DeviceId stage_device = node->device.is_valid() ? node->device : node->stage->device();
                LOG_INFO("[VRAM_TRACE] stage.before index=" << i
                                                            << " name=" << node->name
                                                            << " type=" << computeStageTypeName(node->stage->type())
                                                            << " device=" << stage_device.toString());
            }

            if (!runStage(*node, ctx, policy, is_coll))
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage failed: " << node->name);
                notifyStageFailure(node->name, "stage execution returned false");
                return false;
            }

            if (debugEnv().vram_trace && ctx->isGPU())
            {
                const DeviceId stage_device = node->device.is_valid() ? node->device : node->stage->device();
                LOG_INFO("[VRAM_TRACE] stage.after index=" << i
                                                           << " name=" << node->name
                                                           << " type=" << computeStageTypeName(node->stage->type())
                                                           << " device=" << stage_device.toString());
            }

            if (ctx->isGPU() && debugEnv().runtime_debug.sync_after_stage)
            {
                LOG_DEBUG("[DeviceGraphExecutor] sync after stage: " << node->name);
                ctx->synchronize();
            }

            // Mark stage completed for graph dependency tracking
            graph.markCompleted(node->name);

            if (timeline_active && timeline_gpu_ctx)
                stage_timeline_.recordStop(i, timeline_gpu_ctx, timeline_stream);
        }

        // =====================================================================
        // Post-loop: async dump wait + total stats
        // =====================================================================
        if (policy.stage_dump && AsyncStageDumper::isInitialized())
        {
            size_t pending = AsyncStageDumper::pendingTasks();
            if (pending > 0)
            {
                LOG_DEBUG("[DeviceGraphExecutor] Waiting for " << pending << " pending async dumps...");
                AsyncStageDumper::waitForCompletion();
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        if (policy.profiling)
        {
            LOG_DEBUG("[DeviceGraphExecutor] Total execution: " << total_ms << "ms for "
                                                                << schedule.size() << " stages ("
                                                                << (total_ms / schedule.size()) << "ms/stage avg)");

            stats_.total_time_ms += total_ms;
            if (!config_.enable_profiling)
                stats_.total_stages_executed += schedule.size();
            stats_.total_flops += graph.totalEstimatedFlops();
        }

        // After first successful pass, all weight tensors are confirmed on-device.
        // Subsequent graph rebuilds (e.g., after clear_cache) skip per-node
        // weight coherence since weight data never moves between forwards.
        if (!weights_session_cohered_)
            weights_session_cohered_ = true;

        return true;
    }

    bool DeviceGraphExecutor::runStage(
        ComputeNode &node,
        IDeviceContext *ctx,
        const StageRunPolicy &policy,
        bool is_collective)
    {
        if (!node.stage)
        {
            LOG_ERROR("[DeviceGraphExecutor] Node '" << node.name << "' has no stage");
            return false;
        }

        // =====================================================================
        // Transfer Profiling: per-stage H2D/D2H transfer tracking
        // =====================================================================
        std::optional<TransferProfiler::StageScope> transfer_scope;
        if (policy.profiling)
            transfer_scope.emplace(node.name);

        // =====================================================================
        // Collective Stage Intercept
        // =====================================================================
        if (policy.collective_intercept && is_collective)
        {
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
                    if (debugEnv().execution.gpu_graph_collective_segmented)
                    {
                        LOG_DEBUG("[DeviceGraphExecutor] Skipping strided ALLGATHER intercept in segmented collective mode for '" << node.name << "'");
                    }
                    else
                    {
                        LOG_DEBUG("[DeviceGraphExecutor] Attempting strided ALLGATHER intercept for '" << node.name << "'");
                        if (executeCollectiveStridedAllgather(node, ctx))
                            return true;
                        LOG_DEBUG("[DeviceGraphExecutor] Strided ALLGATHER not available, using stage execution");
                    }
                }
            }

            // LOCAL TP / GLOBAL TP: collective_ctx_ is nullptr, stage handles
            // collective internally.  In fast-decode mode (no coherence) we can
            // skip the full coherence/validation pipeline, but we must still
            // accumulate profiling stats so ALLREDUCE/ALLGATHER appear in the
            // executor timing tables.
            if (!policy.coherence)
            {
                ensureStageGPUStreamBound(node, ctx);

                const bool profiling_fast = policy.profiling && config_.enable_profiling;
                std::chrono::high_resolution_clock::time_point t0{};
                if (profiling_fast)
                    t0 = std::chrono::high_resolution_clock::now();

                bool ok = node.stage->execute(ctx);

                if (profiling_fast)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double exec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                    const std::string stage_type_name = computeStageTypeName(node.stage->type());
                    stats_.stage_times_ms[node.name] = exec_ms;
                    stats_.total_execute_ms += exec_ms;
                    stats_.total_stages_executed++;
                    stats_.stage_type_execute_ms[stage_type_name] += exec_ms;
                    stats_.stage_type_counts[stage_type_name]++;

                    const auto stype = node.stage->type();
                    if (stype == ComputeStageType::ALLREDUCE ||
                        stype == ComputeStageType::ALLGATHER ||
                        stype == ComputeStageType::ALLGATHER_V)
                    {
                        stats_.total_collective_ms += exec_ms;
                        stats_.total_collective_calls++;
                    }

                    const auto phase = GraphExecutorStats::currentPhase();
                    PhaseStats *phase_stats = nullptr;
                    if (phase == ExecutionPhase::PREFILL)
                        phase_stats = &stats_.prefill;
                    else if (phase == ExecutionPhase::DECODE)
                        phase_stats = &stats_.decode;

                    if (phase_stats)
                    {
                        phase_stats->total_execute_ms += exec_ms;
                        phase_stats->total_stages_executed++;
                        phase_stats->stage_type_execute_ms[stage_type_name] += exec_ms;
                        phase_stats->stage_type_counts[stage_type_name]++;
                        if (stype == ComputeStageType::ALLREDUCE ||
                            stype == ComputeStageType::ALLGATHER ||
                            stype == ComputeStageType::ALLGATHER_V)
                        {
                            phase_stats->total_collective_ms += exec_ms;
                            phase_stats->total_collective_calls++;
                        }
                    }
                }

                return ok;
            }
        }

        // =====================================================================
        // Profiling phase breakdown setup
        // =====================================================================
        const bool profiling = policy.profiling && config_.enable_profiling;
        const int layer_idx = config_.current_layer_idx;

        std::chrono::high_resolution_clock::time_point phase_start{}, phase_end{};
        double input_cohere_ms = 0.0, weight_cohere_ms = 0.0, output_alloc_ms = 0.0;
        double dump_input_ms = 0.0, execute_ms = 0.0, mark_dirty_ms = 0.0;
        double get_dump_info_ms = 0.0, dump_output_ms = 0.0, verify_ms = 0.0, callback_ms = 0.0;

        // =====================================================================
        // getDumpInfo caching (needed by coherence, dumps, validation, callback)
        // Skipped entirely in fast decode — zero overhead.
        // =====================================================================
        const bool need_dump_info = policy.coherence || policy.stage_dump ||
                                    policy.pointer_validation || policy.snapshot_callback ||
                                    (arena_ != nullptr);
        StageDumpInfo empty_dump_info{};
        const StageDumpInfo *dump_info_ptr = &empty_dump_info;
        if (need_dump_info)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            dump_info_ptr = &node.stage->getDumpInfo();
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                get_dump_info_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }
        const StageDumpInfo &cached_dump_info = *dump_info_ptr;

        // =====================================================================
        // Stage Coherence: arena contract input/output + weight uploads
        // =====================================================================
        // Contract is ALWAYS fetched when we have an arena. mark_dirty (which
        // uses the contract) is unconditional because marking outputs
        // DEVICE_AUTHORITATIVE is a correctness requirement, not overhead.
        // The coherence flag controls whether prepareForRead/Write are called.
        const StageBufferContract contract = arena_ ? node.stage->bufferContract() : StageBufferContract{};
        const bool use_contract = !contract.empty() && arena_ != nullptr;

        // Bind GPU stream early so coherence operations (H2D/D2H) run on
        // the same stream as the stage's compute kernels.
        ensureStageGPUStreamBound(node, ctx);
        void *stage_stream = node.stage ? node.stage->gpuStream() : nullptr;

        if (policy.coherence)
        {
            auto coh_policy = node.stage->coherencePolicy();
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();

            LOG_DEBUG("[DeviceGraphExecutor] Stage '" << node.name << "' coherencePolicy=" << toString(coh_policy)
                                                      << " target_device=" << target_device.to_string()
                                                      << " use_contract=" << use_contract);

            if (use_contract)
            {
                // Contract-based input coherence
                if (profiling)
                    phase_start = std::chrono::high_resolution_clock::now();

                for (const auto &binding : contract.allArenaReads())
                {
                    if (!arena_->prepareForRead(binding.id, target_device, stage_stream))
                    {
                        LOG_ERROR("[DeviceGraphExecutor] Arena prepareForRead failed for "
                                  << bufferIdName(binding.id) << " in stage '" << node.name << "'");
                        return false;
                    }
                }

                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    input_cohere_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }

                // Weight coherence (not arena-managed, use direct ensureOnDevice).
                // Only processes contract.weight_tensors — the canonical weight list.
                // dump_info.weights duplicates these, and dump_info.inputs are activation
                // tensors already handled by arena coherence above.
                //
                // Per-node fast path: a cached graph node only needs weight
                // coherence once. Do not use a session-wide shortcut here:
                // prefill and decode use distinct cached graph shapes, so a
                // later decode graph may contain nodes whose weights have never
                // been prepared for this device.
                if (policy.weight_coherence && !node.weights_cohered)
                {
                    if (profiling)
                        phase_start = std::chrono::high_resolution_clock::now();

                    for (auto *weight : contract.weight_tensors)
                    {
                        if (auto *tb = dynamic_cast<TensorBase *>(weight))
                        {
                            // Skip weights whose GEMM representation is managed by the
                            // PreparedWeightStore-backed GPU pipeline.
                            // The kernel owns its device copy in pooled VRAM — calling
                            // ensureOnDevice would fail because the host data may have
                            // been released after pipeline upload.
                            if (tb->hasPreparedDeviceState())
                            {
                                continue;
                            }
                            if (!tb->ensureOnDevice(target_device, stage_stream))
                            {
                                LOG_ERROR("[DeviceGraphExecutor] Weight upload failed for stage '"
                                          << node.name << "'"
                                          << " tensor=" << static_cast<void *>(tb)
                                          << " name=" << (tb->debugName().empty() ? "(unnamed)" : tb->debugName())
                                          << " shape=[" << tb->rows() << "," << tb->cols() << "]"
                                          << " type=" << static_cast<int>(tb->native_type())
                                          << " device=" << target_device.toString());
                                return false;
                            }
                        }
                    }

                    if (profiling)
                    {
                        phase_end = std::chrono::high_resolution_clock::now();
                        weight_cohere_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                    }
                    node.weights_cohered = true;
                }

                // Output coherence (arena writes)
                if (profiling)
                    phase_start = std::chrono::high_resolution_clock::now();

                for (const auto &binding : contract.allWrites())
                {
                    if (!arena_->prepareForWrite(binding.id, target_device, stage_stream))
                    {
                        LOG_ERROR("[DeviceGraphExecutor] Arena prepareForWrite failed for "
                                  << bufferIdName(binding.id) << " in stage '" << node.name << "'");
                        return false;
                    }
                }

                if (profiling)
                {
                    phase_end = std::chrono::high_resolution_clock::now();
                    output_alloc_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
                }
            }
            else if (coh_policy != CoherencePolicy::NONE)
            {
                if (!target_device.is_cpu())
                {
                    LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                              << "' has coherencePolicy=" << toString(coh_policy)
                                                              << " but no BufferArena + contract. All GPU stages must implement bufferContract().");
                    return false;
                }
            }
        }

        // =====================================================================
        // ENTRY Verification (Debug/Integration only)
        // =====================================================================
        std::string prepared_error;
        if (!node.stage->validatePreparedWeights(&prepared_error))
        {
            LOG_ERROR("[DeviceGraphExecutor] Prepared weight validation failed for stage '"
                      << node.name << "': " << prepared_error);
            return false;
        }

#if LLAMINAR_ASSERTIONS_ACTIVE
        if (policy.validation && debugEnv().validation.validate_inputs)
        {
            verifyStageEntry(node, layer_idx);
        }
#endif

        // =====================================================================
        // Stage Dump: input snapshots
        // =====================================================================
        StageDumpContext dump_ctx;
        const bool should_dump = policy.stage_dump && StageDumper::shouldDump(
                                                          node.stage.get(),
                                                          node.name,
                                                          config_.current_layer_idx,
                                                          config_.current_iteration,
                                                          config_.mpi_rank);

        if (should_dump)
        {
            const auto &dump_cfg = debugEnv().stage_dump;
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            dump_ctx = StageDumper::beginDump(
                node.stage.get(),
                node.name,
                config_.current_layer_idx,
                config_.current_iteration,
                config_.mpi_rank);

            if (dump_cfg.async_dump)
            {
                if (!AsyncStageDumper::isInitialized())
                    AsyncStageDumper::initialize(dump_cfg.async_threads);
                AsyncStageDumper::enqueueInputs(dump_ctx, cached_dump_info);
            }
            else
            {
                StageDumper::dumpInputs(dump_ctx, node.stage.get());
            }
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                dump_input_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        // =====================================================================
        // GPU Pointer Validation
        // =====================================================================
        if (policy.pointer_validation && debugEnv().validation.validate_gpu_ptrs)
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            if (target_device.is_gpu())
            {
                const int expected_ordinal = target_device.toKernelDeviceIndex();
                auto *gpu_ctx = tryGetWorkerContext(target_device);
                bool ptr_validation_failed = false;
                auto validatePtr = [&](const char *label, const char *tensor_name, ITensor *tensor)
                {
                    if (!validateStagePointerSet(
                            gpu_ctx, node.name, label, expected_ordinal,
                            tensor, tensor_name, /*dump_pointer_events=*/true))
                    {
                        ptr_validation_failed = true;
                    }
                };
                for (const auto &input : cached_dump_info.inputs)
                    validatePtr("input", input.name ? input.name : "(unnamed)", const_cast<ITensor *>(input.tensor));
                for (const auto &output : cached_dump_info.outputs)
                    validatePtr("output", output.name ? output.name : "(unnamed)", output.tensor);
                for (const auto &weight : cached_dump_info.weights)
                    validatePtr("weight", weight.name ? weight.name : "(unnamed)", const_cast<ITensor *>(weight.tensor));

                if (ptr_validation_failed)
                {
                    LOG_ERROR("[GPU_PTR_VIOLATION_ABORT] Aborting stage execute: stage='"
                              << node.name << "' target=" << target_device.to_string()
                              << " expected_ordinal=" << expected_ordinal);
                    return false;
                }
            }
        }

        // =====================================================================
        // EXECUTE
        // =====================================================================
        if (profiling)
            phase_start = std::chrono::high_resolution_clock::now();

        // Stream already bound above (before coherence section)
        bool success = node.stage->execute(ctx);

        if (success && debugEnv().validation.sync_each_stage)
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            if (target_device.is_gpu())
            {
                if (auto *gpu_ctx = tryGetWorkerContext(target_device); gpu_ctx && !gpu_ctx->debugSynchronize())
                {
                    LOG_ERROR("[SYNC_EACH_STAGE] stage='" << node.name
                                                          << "' device=" << target_device.to_string()
                                                          << " device debug synchronization failed");
                    success = false;
                }
            }
        }

        if (profiling)
        {
            phase_end = std::chrono::high_resolution_clock::now();
            execute_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
        }

        // =====================================================================
        // Mark Outputs Dirty (contract-based) — UNCONDITIONAL when arena is
        // present. Skipping this causes stale host reads on the next iteration.
        // =====================================================================
        if (success && use_contract)
        {
            auto coh_policy = node.stage->coherencePolicy();
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();

            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();

            const bool need_event = node.is_final_output || (policy.snapshot_callback && config_.snapshot_callback)
#if LLAMINAR_ASSERTIONS_ACTIVE
                                    || debugEnv().validation.validate_buffers
#endif
                ;

            for (const auto &binding : contract.allWrites())
            {
                if (need_event)
                    arena_->markWritten(binding.id, target_device, node.stage->gpuStream());
                else
                    arena_->markWrittenFlagsOnly(binding.id, target_device);
            }

            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                mark_dirty_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }

            logWatchedPointerProducer(
                node.name,
                cached_dump_info,
                tryGetWorkerContext(node.device.is_valid() ? node.device : node.stage->device()));
            printStageOutputs(node.name, cached_dump_info);
        }

        // =====================================================================
        // Stage Dump: output snapshots
        // =====================================================================
        if (should_dump && success)
        {
            const auto &dump_cfg = debugEnv().stage_dump;
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();

            // Rebuild post-execute dump info before output dumping. Some stages
            // populate diagnostic outputs during execute(), and async dumping
            // consumes the StageDumpInfo passed here rather than re-fetching
            // through StageDumper::dumpOutputs().
            node.stage->invalidateDumpInfoCache();
            const StageDumpInfo &output_dump_info = node.stage->getDumpInfo();

            if (dump_cfg.async_dump)
            {
                AsyncStageDumper::enqueueOutputs(dump_ctx, output_dump_info);
            }
            else
            {
                StageDumper::dumpOutputs(dump_ctx, node.stage.get());
                StageDumper::finalizeDump(dump_ctx, execute_ms);
            }
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                dump_output_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        // =====================================================================
        // EXIT Verification (Debug/Integration only)
        // =====================================================================
#if LLAMINAR_ASSERTIONS_ACTIVE
        if (success && policy.validation && debugEnv().validation.validate_buffers)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            verifyStageExit(node, layer_idx);
            success = validateStageOutputs(node);
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                verify_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }
#endif

        // =====================================================================
        // Snapshot Callback
        // =====================================================================
        if (success && policy.snapshot_callback && config_.snapshot_callback)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            // Re-fetch dump info post-execute: stages like MoEExpertComputeStage stash
            // runtime routing data during execute() and invalidate the cache.
            // Since cached_dump_info aliases the stage's internal cached object,
            // this getDumpInfo() call rebuilds it with post-execute data.
            node.stage->getDumpInfo();
            cached_dump_info.ensureOutputsOnHost();
            LOG_DEBUG("[DeviceGraphExecutor::runStage] Invoking callback for " << node.name);
            config_.snapshot_callback(node.name, cached_dump_info);
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                callback_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        // =====================================================================
        // Profiling Stats
        // =====================================================================
        if (profiling)
        {
            double total_overhead_ms = input_cohere_ms + weight_cohere_ms + output_alloc_ms +
                                       dump_input_ms + mark_dirty_ms + dump_output_ms +
                                       verify_ms + callback_ms + get_dump_info_ms;
            double total_ms = total_overhead_ms + execute_ms;

            if (total_ms > 1.0 || input_cohere_ms > 0.5 || weight_cohere_ms > 0.5 ||
                output_alloc_ms > 0.5 || execute_ms > 0.5 || verify_ms > 0.5 || callback_ms > 0.5)
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
                                                           << " total=" << total_ms << "ms");
            }

            stats_.stage_times_ms[node.name] = total_ms;
            stats_.total_execute_ms += execute_ms;
            stats_.total_stages_executed++;
            const std::string stage_type_name = computeStageTypeName(node.stage->type());
            stats_.stage_type_execute_ms[stage_type_name] += execute_ms;
            stats_.stage_type_counts[stage_type_name]++;

            const auto stype = node.stage->type();
            if (stype == ComputeStageType::ALLREDUCE ||
                stype == ComputeStageType::ALLGATHER ||
                stype == ComputeStageType::ALLGATHER_V)
            {
                stats_.total_collective_ms += execute_ms;
                stats_.total_collective_calls++;
            }

            stats_.overhead.input_cohere_ms += input_cohere_ms;
            stats_.overhead.weight_cohere_ms += weight_cohere_ms;
            stats_.overhead.output_alloc_ms += output_alloc_ms;
            stats_.overhead.mark_dirty_ms += mark_dirty_ms;
            stats_.overhead.dump_input_ms += dump_input_ms;
            stats_.overhead.dump_output_ms += dump_output_ms;
            stats_.overhead.verify_ms += verify_ms;
            stats_.overhead.callback_ms += callback_ms;
            stats_.overhead.get_dump_info_ms += get_dump_info_ms;

            const auto phase = GraphExecutorStats::currentPhase();
            PhaseStats *phase_stats = nullptr;
            if (phase == ExecutionPhase::PREFILL)
                phase_stats = &stats_.prefill;
            else if (phase == ExecutionPhase::DECODE)
                phase_stats = &stats_.decode;

            if (phase_stats)
            {
                phase_stats->total_execute_ms += execute_ms;
                phase_stats->total_stages_executed++;
                phase_stats->stage_type_execute_ms[stage_type_name] += execute_ms;
                phase_stats->stage_type_counts[stage_type_name]++;
                if (stype == ComputeStageType::ALLREDUCE ||
                    stype == ComputeStageType::ALLGATHER ||
                    stype == ComputeStageType::ALLGATHER_V)
                {
                    phase_stats->total_collective_ms += execute_ms;
                    phase_stats->total_collective_calls++;
                }
                phase_stats->overhead.input_cohere_ms += input_cohere_ms;
                phase_stats->overhead.weight_cohere_ms += weight_cohere_ms;
                phase_stats->overhead.output_alloc_ms += output_alloc_ms;
                phase_stats->overhead.mark_dirty_ms += mark_dirty_ms;
                phase_stats->overhead.dump_input_ms += dump_input_ms;
                phase_stats->overhead.dump_output_ms += dump_output_ms;
                phase_stats->overhead.verify_ms += verify_ms;
                phase_stats->overhead.callback_ms += callback_ms;
                phase_stats->overhead.get_dump_info_ms += get_dump_info_ms;
            }

            LOG_DEBUG("[DeviceGraphExecutor] Stage '" << node.name << "' took " << total_ms << " ms (execute=" << execute_ms << "ms, overhead=" << total_overhead_ms << "ms)");
        }

        return success;
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

            if (cancellationRequested(name))
                return false;

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
                notifyStageFailure(name, "stage execution returned false");
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

            // Use stream directly with LOG_DEBUG
            LOG_DEBUG(header.str() << first_row.str() << last_row.str());
        }
    }

    static void logWatchedPointerProducer(
        const std::string &stage_name,
        const StageDumpInfo &dump_info,
        const IWorkerGPUContext *gpu_ctx)
    {
        const auto &validation = debugEnv().validation;
        if (!validation.trace_local_tp_pointer || !gpu_ctx)
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

            const auto info = gpu_ctx->inspectPointer(gpu_ptr);
            if (!info.known || !info.active || !info.base_ptr || info.size_bytes == 0)
            {
                continue;
            }

            const uintptr_t begin = reinterpret_cast<uintptr_t>(info.base_ptr);
            const uintptr_t end = begin + info.size_bytes;
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
                     << " owner_base=" << info.base_ptr
                     << " owner_bytes=" << info.size_bytes
                     << " owner_device=" << info.actual_device
                     << " owner_seq=" << info.sequence
                     << " owner_thread=" << info.thread_hash
                     << " offset=" << offset
                     << " tensor=" << static_cast<void *>(tb)
                     << " tensor_name=" << (tb->debugName().empty() ? "(unnamed)" : tb->debugName()));
        }
    }

    bool DeviceGraphExecutor::executeNode(ComputeNode &node, IDeviceContext *ctx)
    {
        // Legacy entry point — delegates to unified runStage with full policy.
        // Retained for backward compatibility (used by executeMultiDevice and
        // graph capture's non-captured segment execution).
        const bool is_collective = node.stage &&
                                   (node.stage->type() == ComputeStageType::ALLREDUCE ||
                                    node.stage->type() == ComputeStageType::ALLGATHER ||
                                    node.stage->type() == ComputeStageType::ALLGATHER_V);
        return runStage(node, ctx, StageRunPolicy::full(), is_collective);
    }

    // =============================================================================
    // Buffer Validation (Debug/Integration Builds Only)
    // Delegated to free functions in StageVerifier.h/.cpp
    // =============================================================================

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
            stats_.total_execute_ms += ms;
            stats_.total_collective_ms += ms;
            stats_.total_collective_calls++;
            stats_.total_stages_executed++;
            const std::string stage_type_name = computeStageTypeName(node.stage->type());
            stats_.stage_type_execute_ms[stage_type_name] += ms;
            stats_.stage_type_counts[stage_type_name]++;

            // Phase-split accumulation
            const auto phase = GraphExecutorStats::currentPhase();
            PhaseStats *phase_stats = nullptr;
            if (phase == ExecutionPhase::PREFILL)
                phase_stats = &stats_.prefill;
            else if (phase == ExecutionPhase::DECODE)
                phase_stats = &stats_.decode;
            if (phase_stats)
            {
                phase_stats->total_execute_ms += ms;
                phase_stats->total_stages_executed++;
                phase_stats->total_collective_ms += ms;
                phase_stats->total_collective_calls++;
                phase_stats->stage_type_execute_ms[stage_type_name] += ms;
                phase_stats->stage_type_counts[stage_type_name]++;
            }

            LOG_DEBUG("[DeviceGraphExecutor] ALLREDUCE '" << node.name << "' via CollectiveContext took " << ms << " ms");
        }

        // Record to KernelProfiler so allreduce appears in kernel timing summaries
        if (KernelProfiler::isEnabled())
        {
            uint64_t ns = static_cast<uint64_t>(ms * 1'000'000.0);
            KernelProfiler::record(KernelType::ALLREDUCE, ns);
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
            stats_.total_execute_ms += ms;
            stats_.total_collective_ms += ms;
            stats_.total_collective_calls++;
            stats_.total_stages_executed++;
            const std::string stage_type_name = computeStageTypeName(node.stage->type());
            stats_.stage_type_execute_ms[stage_type_name] += ms;
            stats_.stage_type_counts[stage_type_name]++;

            // Phase-split accumulation
            const auto phase = GraphExecutorStats::currentPhase();
            PhaseStats *phase_stats = nullptr;
            if (phase == ExecutionPhase::PREFILL)
                phase_stats = &stats_.prefill;
            else if (phase == ExecutionPhase::DECODE)
                phase_stats = &stats_.decode;
            if (phase_stats)
            {
                phase_stats->total_execute_ms += ms;
                phase_stats->total_stages_executed++;
                phase_stats->total_collective_ms += ms;
                phase_stats->total_collective_calls++;
                phase_stats->stage_type_execute_ms[stage_type_name] += ms;
                phase_stats->stage_type_counts[stage_type_name]++;
            }

            LOG_DEBUG("[DeviceGraphExecutor] ALLGATHER '" << node.name << "' via CollectiveContext took " << ms << " ms");
        }

        // Record to KernelProfiler so allgather appears in kernel timing summaries
        if (KernelProfiler::isEnabled())
        {
            uint64_t ns = static_cast<uint64_t>(ms * 1'000'000.0);
            KernelProfiler::record(KernelType::ALLGATHER, ns);
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
                stats_.total_execute_ms += ms;
                stats_.total_collective_ms += ms;
                stats_.total_collective_calls++;
                stats_.total_stages_executed++;
                const std::string stage_type_name = computeStageTypeName(node.stage->type());
                stats_.stage_type_execute_ms[stage_type_name] += ms;
                stats_.stage_type_counts[stage_type_name]++;

                // Phase-split accumulation
                const auto phase = GraphExecutorStats::currentPhase();
                PhaseStats *phase_stats = nullptr;
                if (phase == ExecutionPhase::PREFILL)
                    phase_stats = &stats_.prefill;
                else if (phase == ExecutionPhase::DECODE)
                    phase_stats = &stats_.decode;
                if (phase_stats)
                {
                    phase_stats->total_execute_ms += ms;
                    phase_stats->total_stages_executed++;
                    phase_stats->total_collective_ms += ms;
                    phase_stats->total_collective_calls++;
                    phase_stats->stage_type_execute_ms[stage_type_name] += ms;
                    phase_stats->stage_type_counts[stage_type_name]++;
                }
            }
            // Record to KernelProfiler so strided allgather appears in kernel timing summaries
            if (KernelProfiler::isEnabled())
            {
                uint64_t ns = static_cast<uint64_t>(ms * 1'000'000.0);
                KernelProfiler::record(KernelType::ALLGATHER, ns);
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
