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
#include "../coherence/CoherencePolicy.h"
#include "../collective/CollectiveContext.h"
#include "../../compute_stages/stages/AllreduceStage.h"
#include "../../compute_stages/stages/AllGatherStage.h"
#include "../../../config/TPDomain.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../backends/BackendManager.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../loaders/PreparedWeightStore.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <print>
#include "fort.hpp"
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <typeinfo>

namespace llaminar2
{
    namespace
    {
        IWorkerGPUContext *tryGetWorkerContext(
            const DeviceId &device,
            const GraphExecutorConfig &config)
        {
            if (!device.is_gpu())
            {
                return nullptr;
            }

            try
            {
                if (config.worker_gpu_context_resolver)
                    return config.worker_gpu_context_resolver(device);
                return &GPUDeviceContextPool::instance().getContext(device);
            }
            catch (const std::exception &e)
            {
                LOG_DEBUG("[DeviceGraphExecutor] Failed to resolve worker GPU context for "
                          << device.to_string() << ": " << e.what());
                return nullptr;
            }
        }

        void *resolveWorkerDefaultStream(
            const DeviceId &device,
            const GraphExecutorConfig &config)
        {
            if (auto *gpu_ctx = tryGetWorkerContext(device, config))
            {
                return gpu_ctx->defaultStream();
            }

            return nullptr;
        }

        std::string snapshotOutputName(const StageDumpInfo::OutputBuffer &output)
        {
            return output.name ? output.name : "";
        }

        bool isDeviceTensorSnapshotOutput(const StageDumpInfo::OutputBuffer &output)
        {
            return output.tensor != nullptr;
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

        bool ensureStageGPUStreamBound(
            ComputeNode &node,
            IDeviceContext *ctx,
            const GraphExecutorConfig &config)
        {
            if (!node.stage || node.stage->hasGPUStream())
            {
                return true;
            }

            DeviceId device = node.device.is_valid() ? node.device : node.stage->device();
            if (!device.is_valid() && ctx)
            {
                device = ctx->deviceId();
            }

            if (!device.is_gpu())
            {
                return true;
            }

            void *stream = resolveWorkerDefaultStream(device, config);
            if (stream)
            {
                node.stage->setGPUStream(stream);
                return true;
            }

            LOG_ERROR("[DeviceGraphExecutor] GPU stage '" << node.name
                                                         << "' has no explicit stream for device "
                                                         << device.to_string());
            return false;
        }

        /**
         * @brief Ensure GPU stages have explicit streams for normal eager passes.
         *
         * Null stage streams are dangerous because CUDA/HIP will otherwise fall
         * back to the backend's implicit default stream.  This helper fills
         * missing streams with the owning worker stream while preserving any
         * stream a caller already bound deliberately, such as a graph-capture
         * stream.  Preserving explicit ownership is what keeps handoff rules
         * understandable: callers that bind a stream remain responsible for
         * ordering it, while unbound stages get the executor-owned stream.
         */
        bool bindScheduleToWorkerStreams(
            const std::vector<ComputeGraph::FastScheduleEntry> &schedule,
            IDeviceContext *ctx,
            const GraphExecutorConfig &config)
        {
            if (!ctx || !ctx->isGPU())
            {
                return true;
            }

            for (const auto &entry : schedule)
            {
                auto *node = entry.node;
                if (!node || !node->stage)
                {
                    continue;
                }

                DeviceId device = node->device.is_valid() ? node->device : node->stage->device();
                if (!device.is_valid() && ctx)
                {
                    device = ctx->deviceId();
                }
                if (!device.is_gpu())
                {
                    continue;
                }

                auto *gpu_ctx = tryGetWorkerContext(device, config);
                if (!gpu_ctx)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Could not resolve GPU context while binding stage '"
                              << node->name << "' to " << device.to_string());
                    return false;
                }

                void *worker_stream = gpu_ctx->defaultStream();
                if (!worker_stream)
                {
                    LOG_ERROR("[DeviceGraphExecutor] GPU context for " << device.to_string()
                                                                       << " has no explicit worker stream");
                    return false;
                }

                if (node->stage->hasGPUStream())
                {
                    continue;
                }

                node->stage->setGPUStream(worker_stream);
            }

            return true;
        }

        bool contractReadsNeedTransfer(
            BufferArena *arena,
            const StageBufferContract &contract,
            DeviceId target_device)
        {
            if (!arena || contract.empty())
                return false;

            for (const auto &binding : contract.allArenaReads())
            {
                if (!arena->isRegistered(binding.id))
                    continue;

                auto *tensor = dynamic_cast<TensorBase *>(arena->getTensor(binding.id));
                if (!tensor)
                    continue;

                const CoherenceState state = arena->getCoherenceState(binding.id);
                if (state.needsTransferTo(target_device, tensor->coherenceState()))
                    return true;
            }

            return false;
        }

        /**
         * @brief Report whether a fast-policy stage still needs output storage.
         *
         * Fast decode deliberately skips the full coherence walk after graph
         * addresses have stabilized.  That optimization is legal only after
         * every declared write already owns storage on the stage's target GPU.
         * A read-resident stage may still have a cold output, especially for a
         * graph family materialized before its first execution.  Checking both
         * the pointer and its device owner makes that first-write allocation an
         * explicit contract condition without adding prepareForWrite calls to
         * steady-state replay.
         */
        bool contractWritesNeedStorage(
            BufferArena *arena,
            const StageBufferContract &contract,
            DeviceId target_device)
        {
            if (!arena || contract.empty() || !target_device.is_gpu())
                return false;

            for (const auto &binding : contract.writesRequiringPrepare())
            {
                if (!arena->isRegistered(binding.id))
                    continue;

                auto *tensor =
                    dynamic_cast<TensorBase *>(arena->getTensor(binding.id));
                if (!tensor)
                    continue;

                const auto resident_device = tensor->current_device();
                if (!tensor->gpu_data_ptr() || !resident_device ||
                    *resident_device != target_device)
                {
                    return true;
                }
            }

            return false;
        }

        bool fastPolicyRequiresContractCoherence(
            const StageRunPolicy &policy,
            BufferArena *arena,
            const StageBufferContract &contract,
            DeviceId target_device)
        {
            if (policy.coherence || !arena || contract.empty())
                return false;

            // Host-staged graph-native collectives must always honor arena
            // contracts in decode: they are exactly the CPU bridges between
            // device-resident graph stages. GPU stages also need coherence when
            // a prior host bridge made an input CPU-authoritative or when a
            // newly materialized graph-family output has not received its
            // stable target-device allocation yet.
            return target_device.is_cpu() ||
                   contractReadsNeedTransfer(arena, contract, target_device) ||
                   contractWritesNeedStorage(arena, contract, target_device);
        }
    }

    // =========================================================================
    // GraphSegmentCache & GPU graph capture implementations moved to
    // DeviceGraphExecutor_GraphCapture.cpp
    // =========================================================================

    // Forward declarations for static helpers used by runStage()
    static bool stageChecksumTraceEnabled();
    static bool stageChecksumTraceMatches(const std::string &stage_name);
    static void printStageOutputs(const std::string &stage_name, const StageDumpInfo &dump_info, void *stream);
    static void traceStageOutputChecksums(
        const std::string &stage_name,
        const IComputeStage *stage,
        const StageDumpInfo &dump_info,
        void *stream);
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
        printPhaseTable("HOST EXECUTOR STAGE PROFILING — PREFILL", prefill, prefill_tokens);
        printPhaseTable("HOST EXECUTOR STAGE PROFILING — DECODE", decode, decode_tokens);

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

    void GraphExecutorStats::recordPerfStats(const std::string &device_name) const
    {
        if (!PerfStatsCollector::isEnabled())
            return;

        auto record_phase = [&](const char *phase_name, const PhaseStats &phase)
        {
            if (!phase_name || phase.total_stages_executed == 0)
                return;

            const PerfStatsCollector::Tags base_tags{
                {"attribution", "host"},
                {"source", "device_graph_executor"},
                {"graph_capture_scope", "eager_or_capture_setup"},
                {"note", "host_executor_timing_not_gpu_stage_time"}};

            PerfStatsCollector::recordTimingNs(
                "stage_executor_cpu",
                "execute_total",
                static_cast<uint64_t>(phase.total_execute_ms * 1.0e6),
                phase_name,
                device_name,
                base_tags);

            if (phase.total_collective_ms > 0.0)
            {
                auto collective_tags = base_tags;
                collective_tags.emplace("calls", std::to_string(phase.total_collective_calls));
                PerfStatsCollector::recordTimingNs(
                    "stage_executor_cpu",
                    "collective_total",
                    static_cast<uint64_t>(phase.total_collective_ms * 1.0e6),
                    phase_name,
                    device_name,
                    std::move(collective_tags));
            }

            const double overhead_ms = phase.overhead.total();
            if (overhead_ms > 0.0)
            {
                PerfStatsCollector::recordTimingNs(
                    "stage_executor_cpu",
                    "overhead_total",
                    static_cast<uint64_t>(overhead_ms * 1.0e6),
                    phase_name,
                    device_name,
                    base_tags);
            }

            for (const auto &[stage_type, ms] : phase.stage_type_execute_ms)
            {
                auto tags = base_tags;
                const auto count_it = phase.stage_type_counts.find(stage_type);
                tags.emplace("stage_type", stage_type);
                tags.emplace("stage_count",
                             count_it == phase.stage_type_counts.end()
                                 ? "0"
                                 : std::to_string(count_it->second));
                PerfStatsCollector::recordTimingNs(
                    "stage_executor_cpu",
                    std::string("type.") + stage_type,
                    static_cast<uint64_t>(ms * 1.0e6),
                    phase_name,
                    device_name,
                    std::move(tags));
            }
        };

        record_phase("prefill", prefill);
        record_phase("decode", decode);
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

    bool DeviceGraphExecutor::executeWithSnapshotManifest(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        GraphSnapshotManifest &snapshot_manifest)
    {
        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphExecutor] Null device context");
            return false;
        }
        if (graph.size() == 0)
            return true;

        graph.reset();
        switch (config_.mode)
        {
        case ExecutionMode::SEQUENTIAL:
        case ExecutionMode::PARALLEL:
            return runStages(
                graph,
                ctx,
                StageRunPolicy::full(),
                nullptr,
                &snapshot_manifest);
        case ExecutionMode::PIPELINED:
            LOG_WARN("[DeviceGraphExecutor] Pipelined mode not yet implemented, using sequential");
            return runStages(
                graph,
                ctx,
                StageRunPolicy::full(),
                nullptr,
                &snapshot_manifest);
        default:
            LOG_ERROR("[DeviceGraphExecutor] Unknown execution mode");
            return false;
        }
    }

    bool DeviceGraphExecutor::executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                                                const std::unordered_set<std::string> *collective_nodes,
                                                GraphSnapshotManifest *snapshot_manifest)
    {
        StageRunPolicy policy = StageRunPolicy::fastDecode();
        /*
         * Eager fast decode has no captured D2D snapshot nodes.  When a parity
         * diagnostic callback is armed, publish each stage immediately while
         * its arena output still contains that stage's bytes.  Deferring these
         * callbacks until the complete graph returns lets normal buffer aliasing
         * overwrite early values such as EMBEDDING with a later residual.
         */
        policy.snapshot_callback = config_.snapshot_callback != nullptr;
        return runStages(
            graph,
            ctx,
            policy,
            collective_nodes,
            snapshot_manifest);
    }

    bool DeviceGraphExecutor::prepareStageArenaFrontierForCapture(
        ComputeNode &node,
        const std::vector<BufferBinding> &external_reads,
        DeviceId capture_device,
        void *capture_stream,
        std::unordered_set<ITensor *> &prepared_inputs,
        std::unordered_set<ITensor *> &prepared_outputs,
        const char *context)
    {
        if (!arena_ || !node.stage)
        {
            LOG_ERROR("[DeviceGraphExecutor] Stage arena capture frontier requires an arena and stage"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }
        if (!capture_device.is_gpu() || !capture_stream)
        {
            LOG_ERROR("[DeviceGraphExecutor] Stage arena capture frontier requires a GPU and exact stream"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[DeviceGraphExecutor] Stage arena frontier must be prepared before beginCapture()"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }

        DeviceId target_device =
            node.device.is_valid() ? node.device : node.stage->device();
        if (!target_device.is_valid())
            target_device = capture_device;
        if (target_device != capture_device)
        {
            LOG_ERROR("[DeviceGraphExecutor] A native graph capture cannot span "
                      << capture_device.toString() << " and "
                      << target_device.toString() << " at stage '" << node.name << "'"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }

        try
        {
            for (const auto &binding : external_reads)
            {
                ITensor *tensor = arena_->getTensor(binding.id);
                if (!tensor)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Graph input "
                              << bufferIdName(binding.id)
                              << " is not bound for stage '" << node.name << "'"
                              << (context ? std::string(" (") + context + ")" : std::string()));
                    return false;
                }
                if (!prepared_inputs.insert(tensor).second)
                    continue;

                TransferEngine::requireDeviceInput(
                    tensor,
                    capture_device,
                    capture_stream);
            }

            const StageBufferContract contract = node.stage->bufferContract();
            for (const auto &binding : contract.writesRequiringPrepare())
            {
                ITensor *tensor = arena_->getTensor(binding.id);
                if (!tensor)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Graph output "
                              << bufferIdName(binding.id)
                              << " is not bound for stage '" << node.name << "'"
                              << (context ? std::string(" (") + context + ")" : std::string()));
                    return false;
                }
                if (!prepared_outputs.insert(tensor).second)
                    continue;

                if (!arena_->prepareForWrite(
                        binding.id,
                        capture_device,
                        capture_stream))
                {
                    LOG_ERROR("[DeviceGraphExecutor] Failed to allocate graph output "
                              << bufferIdName(binding.id)
                              << " before capture at stage '" << node.name << "'"
                              << (context ? std::string(" (") + context + ")" : std::string()));
                    return false;
                }
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceGraphExecutor] Failed to prepare graph arena frontier at stage '"
                      << node.name << "'"
                      << (context ? std::string(" (") + context + ")" : std::string())
                      << ": " << e.what());
            return false;
        }

        return true;
    }

    bool DeviceGraphExecutor::prepareGraphStorageForCapture(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        void *capture_stream,
        const char *context)
    {
        if (!arena_)
        {
            LOG_ERROR("[DeviceGraphExecutor] Cannot prepare graph storage without a BufferArena"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }
        if (!ctx || !ctx->isGPU())
        {
            LOG_ERROR("[DeviceGraphExecutor] Graph storage preparation requires a GPU context"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }
        if (!capture_stream)
        {
            LOG_ERROR("[DeviceGraphExecutor] Graph storage preparation requires the exact capture stream"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }
        if (isGraphCaptureActive())
        {
            LOG_ERROR("[DeviceGraphExecutor] Graph storage must be prepared before beginCapture()"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }

        const DeviceId capture_device = ctx->deviceId();
        std::unordered_set<ITensor *> prepared_inputs;
        std::unordered_set<ITensor *> prepared_outputs;
        GraphArenaDependencyTracker dependency_tracker;

        for (const auto &name : graph.getExecutionOrder())
        {
            ComputeNode *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;

            const StageBufferContract contract = node->stage->bufferContract();
            const auto external_reads =
                dependency_tracker.observeStage(contract);
            if (!prepareStageArenaFrontierForCapture(
                    *node,
                    external_reads,
                    capture_device,
                    capture_stream,
                    prepared_inputs,
                    prepared_outputs,
                    context))
            {
                return false;
            }
        }

        return true;
    }

    bool DeviceGraphExecutor::prepareSnapshotsForGraphCapture(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        void *producer_stream,
        const char *context,
        GraphSnapshotManifest *snapshot_manifest)
    {
        if (!config_.snapshot_callback)
            return true;

        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphExecutor] Cannot prepare graph snapshots without a device context"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }

        DeviceId fallback_device = ctx->deviceId();
        if (!fallback_device.is_gpu())
            return true;

        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphExecutor] Cannot prepare graph snapshots without an explicit GPU stream"
                      << (context ? std::string(" (") + context + ")" : std::string()));
            return false;
        }

        for (const auto &name : graph.getExecutionOrder())
        {
            ComputeNode *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;

            node->stage->setGPUStream(producer_stream);

            DeviceId target_device = node->device.is_valid() ? node->device : node->stage->device();
            if (!target_device.is_valid())
                target_device = fallback_device;

            GraphSnapshotManifest &manifest =
                snapshot_manifest ? *snapshot_manifest
                                  : transient_snapshot_manifest_;
            if (!prepareGraphSnapshotCopies(
                    *node,
                    target_device,
                    producer_stream,
                    manifest))
            {
                LOG_ERROR("[DeviceGraphExecutor] Failed to prepare graph snapshots for stage '"
                          << name << "'"
                          << (context ? std::string(" (") + context + ")" : std::string()));
                return false;
            }
        }

        return true;
    }

    bool DeviceGraphExecutor::shouldCaptureSnapshotStage(
        const std::string &node_name,
        const StageDumpInfo &dump_info) const
    {
        return !config_.snapshot_stage_filter ||
               config_.snapshot_stage_filter(node_name, dump_info);
    }

    bool DeviceGraphExecutor::prepareGraphSnapshotCopies(
        ComputeNode &node,
        DeviceId target_device,
        void *producer_stream,
        GraphSnapshotManifest &snapshot_manifest)
    {
        return prepareOrRecordGraphSnapshotCopies(
            node,
            target_device,
            producer_stream,
            /*record_device_copy=*/false,
            snapshot_manifest);
    }

    bool DeviceGraphExecutor::captureGraphSnapshotCopies(
        ComputeNode &node,
        DeviceId target_device,
        void *producer_stream,
        GraphSnapshotManifest &snapshot_manifest)
    {
        return prepareOrRecordGraphSnapshotCopies(
            node,
            target_device,
            producer_stream,
            /*record_device_copy=*/true,
            snapshot_manifest);
    }

    bool DeviceGraphExecutor::prepareOrRecordGraphSnapshotCopies(
        ComputeNode &node,
        DeviceId target_device,
        void *producer_stream,
        bool record_device_copy,
        GraphSnapshotManifest &snapshot_manifest)
    {
        if (!config_.snapshot_callback || !node.stage)
            return true;
        if (!target_device.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                     << "' cannot "
                                                     << (record_device_copy ? "record" : "prepare")
                                                     << " graph snapshots without an explicit GPU stream");
            return false;
        }

        StageDumpInfo dump_info = node.stage->refreshDumpInfoSnapshot();
        const bool capture_active = isGraphCaptureActive();
        const bool selected =
            shouldCaptureSnapshotStage(node.name, dump_info);
        if (!selected)
        {
            if (capture_active &&
                !snapshot_manifest.filtered_stages.contains(node.name))
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage '"
                          << node.name
                          << "' changed snapshot filter selection during graph capture");
                return false;
            }

            snapshot_manifest.stage_copies.erase(node.name);
            snapshot_manifest.outputless_stages.erase(node.name);
            snapshot_manifest.filtered_stages.insert(node.name);
            return true;
        }

        if (capture_active &&
            snapshot_manifest.filtered_stages.contains(node.name))
        {
            LOG_ERROR("[DeviceGraphExecutor] Stage '"
                      << node.name
                      << "' changed snapshot filter selection during graph capture");
            return false;
        }
        snapshot_manifest.filtered_stages.erase(node.name);

        std::vector<size_t> graph_output_indices;
        graph_output_indices.reserve(dump_info.outputs.size());
        for (size_t i = 0; i < dump_info.outputs.size(); ++i)
        {
            if (isDeviceTensorSnapshotOutput(dump_info.outputs[i]))
                graph_output_indices.push_back(i);
        }

        if (graph_output_indices.empty())
        {
            if (capture_active)
            {
                if (snapshot_manifest.outputless_stages.contains(node.name))
                    return true;

                LOG_ERROR("[DeviceGraphExecutor] GPU snapshot stage '"
                          << node.name
                          << "' lost all tensor-backed outputs during graph capture");
                return false;
            }

            /*
             * Record intentional absence during launch preparation. Capture
             * must observe the same outputless contract; a later appearance is
             * descriptor drift and therefore fatal.
             */
            snapshot_manifest.stage_copies.erase(node.name);
            snapshot_manifest.outputless_stages.insert(node.name);
            return true;
        }

        snapshot_manifest.outputless_stages.erase(node.name);

        auto &stage_copies = snapshot_manifest.stage_copies[node.name];
        if (stage_copies.outputs.size() != graph_output_indices.size())
        {
            if (isGraphCaptureActive())
            {
                std::vector<GraphSnapshotOutputCopy> reconciled;
                reconciled.resize(graph_output_indices.size());
                std::vector<bool> consumed(stage_copies.outputs.size(), false);
                bool matched_all = true;

                for (size_t i = 0; i < graph_output_indices.size(); ++i)
                {
                    const auto &output = dump_info.outputs[graph_output_indices[i]];
                    const std::string output_name = snapshotOutputName(output);
                    size_t match_index = stage_copies.outputs.size();
                    if (!output_name.empty())
                    {
                        for (size_t j = 0; j < stage_copies.outputs.size(); ++j)
                        {
                            if (!consumed[j] && stage_copies.outputs[j].name == output_name)
                            {
                                consumed[j] = true;
                                match_index = j;
                                break;
                            }
                        }
                    }
                    if (match_index == stage_copies.outputs.size())
                    {
                        matched_all = false;
                        break;
                    }
                    reconciled[i] = std::move(stage_copies.outputs[match_index]);
                }

                if (!matched_all)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                             << "' changed snapshot output count during graph capture (old="
                                                             << stage_copies.outputs.size()
                                                             << " new=" << graph_output_indices.size()
                                                             << ") and the device-backed outputs could not be reconciled by name");
                    return false;
                }

                stage_copies.outputs = std::move(reconciled);
            }
            else
            {
                stage_copies.outputs.clear();
                stage_copies.outputs.resize(graph_output_indices.size());
            }
        }

        for (size_t i = 0; i < graph_output_indices.size(); ++i)
        {
            const auto &output = dump_info.outputs[graph_output_indices[i]];
            auto &copy = stage_copies.outputs[i];

            const std::string output_name = snapshotOutputName(output);
            const std::string output_dtype = output.dtype ? output.dtype : "FP32";
            const size_t output_byte_size =
                output.byte_size > 0
                    ? output.byte_size
                    : computeByteSizeForDtype(
                          output_dtype.c_str(), output.rows, output.cols);
            if (output_name.empty() || output_byte_size == 0 || !output.tensor)
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                           << "' has an invalid tensor-backed GPU snapshot descriptor"
                                                           << " name='" << output_name << "'"
                                                           << " rows=" << output.rows
                                                           << " cols=" << output.cols
                                                           << " bytes=" << output_byte_size);
                return false;
            }

            auto *source = dynamic_cast<TensorBase *>(output.tensor);
            const auto source_device = source ? source->current_device() : std::optional<DeviceId>{};
            const void *source_ptr = output.tensor->gpu_data_ptr();
            if ((source_device.has_value() && !source_device->is_gpu()) ||
                !source_ptr)
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                         << "' tensor-backed snapshot output '"
                                                         << (output.name ? output.name : "<unnamed>")
                                                         << "' has no stable GPU source during "
                                                         << (record_device_copy ? "copy recording" : "snapshot preparation"));
                return false;
            }

            const DeviceId copy_device =
                (source_device.has_value() && source_device->is_gpu())
                    ? *source_device
                    : target_device;
            if (!copy_device.is_gpu())
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                         << "' snapshot output '"
                                                         << (output.name ? output.name : "<unnamed>")
                                                         << "' has no GPU source device");
                return false;
            }

            if (capture_active)
            {
                /*
                 * GPU graph snapshot descriptors are immutable capture inputs.
                 * Updating a source pointer or shape after beginCapture() makes
                 * the graph executable and its host publication metadata
                 * describe different tensors. Fail at the recording boundary
                 * instead of silently reconciling the legacy live dump view.
                 */
                const bool descriptor_matches =
                    copy.descriptor_finalized &&
                    copy.name == output_name &&
                    copy.dtype == output_dtype &&
                    copy.rows == output.rows &&
                    copy.cols == output.cols &&
                    copy.element_size == output.element_size &&
                    copy.byte_size == output_byte_size &&
                    copy.device == copy_device &&
                    copy.source_ptr == source_ptr;
                if (!descriptor_matches)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                              << "' changed GPU snapshot descriptor during graph capture"
                                                              << " output='" << output_name << "'"
                                                              << " prepared_source=" << copy.source_ptr
                                                              << " captured_source=" << source_ptr
                                                              << " prepared_dtype=" << copy.dtype
                                                              << " captured_dtype=" << output_dtype
                                                              << " prepared_shape=[" << copy.rows << ',' << copy.cols << ']'
                                                              << " captured_shape=[" << output.rows << ',' << output.cols << ']'
                                                              << " prepared_bytes=" << copy.byte_size
                                                              << " captured_bytes=" << output_byte_size);
                    return false;
                }
            }
            else
            {
                /*
                 * Launch preparation is the sole authority for captured
                 * snapshot topology. Stages must expose their final persistent
                 * output pointer before arithmetic begins; capture later proves
                 * that the executed producer reports the identical descriptor.
                 */
                copy.name = output_name;
                copy.dtype = output_dtype;
                copy.rows = output.rows;
                copy.cols = output.cols;
                copy.element_size = output.element_size;
                copy.byte_size = output_byte_size;
                copy.device = copy_device;
                copy.source_ptr = source_ptr;
                copy.descriptor_finalized = true;
            }

            const size_t storage_elements =
                std::max<size_t>(1, (copy.byte_size + sizeof(float) - 1) / sizeof(float));
            const size_t storage_bytes = storage_elements * sizeof(float);
            const bool needs_new_storage =
                !copy.storage ||
                copy.device != copy_device ||
                copy.storage_bytes < storage_bytes ||
                !copy.storage->isMapped();

            if (needs_new_storage)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[DeviceGraphExecutor] Stage '" << node.name
                                                             << "' needs new graph snapshot storage during graph capture"
                                                             << " for output '" << copy.name << "' bytes="
                                                             << copy.byte_size);
                    return false;
                }

                /*
                 * Snapshot values are intentional debug results surfaced to
                 * the host. Give each immutable graph slot mapped host storage
                 * and record the point-in-time D2D copy into its device-visible
                 * address. This preserves arena-aliased stage values without
                 * reserving one full HBM tensor per output.
                 */
                copy.storage = FP32Tensor::createMapped(
                    std::vector<size_t>{storage_elements},
                    copy_device);
                if (!copy.storage || !copy.storage->isMapped())
                {
                    LOG_ERROR("[DeviceGraphExecutor] Failed to allocate required mapped graph "
                              "snapshot storage for stage '"
                              << node.name << "' output '" << copy.name << "' bytes="
                              << copy.byte_size << " on " << copy_device.toString());
                    return false;
                }
                copy.device = copy_device;
                copy.storage_bytes = storage_bytes;
            }

            if (!copy.storage->gpu_data_ptr())
            {
                LOG_ERROR("[DeviceGraphExecutor] Mapped graph snapshot storage has no device-visible pointer for stage '"
                          << node.name << "' output '" << copy.name << "' on "
                          << copy_device.toString());
                return false;
            }

            if (!record_device_copy)
                continue;

            void *dst_ptr = copy.storage->gpu_data_ptr();
            if (!dst_ptr)
            {
                LOG_ERROR("[DeviceGraphExecutor] Graph snapshot storage has no device pointer for stage '"
                          << node.name << "' output '" << copy.name << "'");
                return false;
            }

            IBackend *backend = getBackendFor(copy_device);
            if (!backend)
            {
                LOG_ERROR("[DeviceGraphExecutor] No backend for graph snapshot copy on "
                          << copy_device.toString());
                return false;
            }

            if (!backend->deviceCopyAsync(
                    dst_ptr,
                    source_ptr,
                    copy.byte_size,
                    copy_device.gpu_ordinal(),
                    producer_stream))
            {
                LOG_ERROR("[DeviceGraphExecutor] Graph snapshot D2D copy failed for stage '"
                          << node.name << "' output '" << copy.name << "' bytes="
                          << copy.byte_size << " device=" << copy_device.toString());
                return false;
            }

            if (isGraphCaptureActive())
            {
                /*
                 * The D2D copy has been recorded as a graph node, but it has
                 * not executed yet.  Recording a normal completion event while
                 * the stream is being captured leaves the tensor with an event
                 * that host publication cannot legally wait on.  The post-graph
                 * publisher re-marks this storage with a real completion event
                 * after the captured graph is launched.
                 */
                TransferEngine::publishGraphOwnedDeviceWrite(
                    copy.storage.get(),
                    copy_device);
            }
            else
            {
                TransferEngine::publishDeviceWrite(
                    copy.storage.get(),
                    copy_device,
                    producer_stream);
            }
        }

        return true;
    }

    bool DeviceGraphExecutor::publishGraphSnapshotCopies(
        const std::string &stage_name,
        void *producer_stream,
        GraphSnapshotManifest &snapshot_manifest)
    {
        if (!config_.snapshot_callback)
            return true;

        /*
         * Snapshot selection is frozen from the concrete output descriptor
         * during graph preflight. Replay publication must not re-enter mutable
         * stage state or repeat a name-only approximation of that decision.
         */
        if (snapshot_manifest.filtered_stages.contains(stage_name))
            return true;

        if (snapshot_manifest.outputless_stages.contains(stage_name))
            return true;

        auto it = snapshot_manifest.stage_copies.find(stage_name);
        if (it == snapshot_manifest.stage_copies.end())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU snapshot stage '" << stage_name
                                                                   << "' has no graph-stable snapshot manifest");
            return false;
        }

        auto &stage_copies = it->second;
        if (stage_copies.outputs.empty())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU snapshot stage '" << stage_name
                                                                   << "' has an empty graph-stable snapshot manifest");
            return false;
        }

        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU snapshot stage '" << stage_name
                                                                   << "' cannot publish without an explicit producer stream");
            return false;
        }

        StageDumpInfo snapshot_info;
        snapshot_info.outputs.reserve(stage_copies.outputs.size());
        for (auto &copy : stage_copies.outputs)
        {
            if (!copy.storage || !copy.device.is_gpu() || copy.byte_size == 0 ||
                copy.name.empty() || copy.dtype.empty())
            {
                LOG_ERROR("[DeviceGraphExecutor] GPU snapshot stage '" << stage_name
                                                                       << "' has an incomplete captured slot for output '"
                                                                       << copy.name << "'");
                return false;
            }

            // Replay updates the mapped storage through a captured D2D node, but
            // no CPU stage code runs then. Mark each slot dirty with a real
            // post-launch event so host publication waits for the just-replayed
            // bytes before reading the host-visible mapping.
            TransferEngine::publishDeviceWrite(
                copy.storage.get(),
                copy.device,
                producer_stream);

            StageDumpInfo::OutputBuffer output;
            output.name = copy.name.c_str();
            output.data = copy.storage->raw_data();
            output.rows = copy.rows;
            output.cols = copy.cols;
            output.dtype = copy.dtype.c_str();
            output.element_size = copy.element_size;
            output.byte_size = copy.byte_size;
            output.tensor = copy.storage.get();
            snapshot_info.outputs.push_back(output);
        }

        snapshot_info.ensureOutputsOnHost(producer_stream);
        config_.snapshot_callback(stage_name, snapshot_info);
        return true;
    }

    bool DeviceGraphExecutor::publishSnapshotsAfterGraphExecution(
        ComputeGraph &graph,
        void *producer_stream_override,
        const char *context,
        GraphSnapshotManifest *snapshot_manifest)
    {
        if (!config_.snapshot_callback)
            return true;

        const auto &order = graph.getExecutionOrder();
        if (order.empty())
            return true;

        auto total_start = std::chrono::high_resolution_clock::now();
        size_t callback_count = 0;

        for (const auto &name : order)
        {
            ComputeNode *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;

            // Captured graph replay/capture launches all recorded work on the
            // graph stream, even if individual stage objects still retain an
            // older worker stream pointer.  When the caller supplies that graph
            // stream, treat it as authoritative so D2H snapshot publication is
            // ordered after the graph launch instead of racing on a stale stream.
            void *producer_stream = producer_stream_override
                                        ? producer_stream_override
                                        : node->stage->gpuStream();
            DeviceId snapshot_device =
                node->device.is_valid() ? node->device : node->stage->device();

            try
            {
                if (snapshot_device.is_gpu())
                {
                    /*
                     * GPU publication is manifest-only. Re-entering the live
                     * stage here revives the pre-graph snapshot system and can
                     * resolve a different cache view, logical row count, or
                     * arena alias than the source pointer baked into the graph.
                     */
                    GraphSnapshotManifest &manifest =
                        snapshot_manifest ? *snapshot_manifest
                                          : transient_snapshot_manifest_;
                    if (!publishGraphSnapshotCopies(
                            name,
                            producer_stream,
                            manifest))
                        return false;
                }
                else
                {
                    // CPU stages execute synchronously and have no captured
                    // snapshot slots, so their ordinary dump path remains the
                    // canonical publication mechanism.
                    StageDumpInfo snapshot_dump_info =
                        node->stage->refreshDumpInfoSnapshot();
                    snapshot_dump_info.ensureOutputsOnHost(producer_stream);
                    config_.snapshot_callback(name, snapshot_dump_info);
                }
                ++callback_count;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[DeviceGraphExecutor] Post-graph snapshot capture failed"
                          << (context ? std::string(" during ") + context : std::string{})
                          << " for stage '" << name << "': " << e.what());
                return false;
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        const double callback_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();
        stats_.overhead.callback_ms += callback_ms;
        switch (GraphExecutorStats::currentPhase())
        {
        case ExecutionPhase::PREFILL:
            stats_.prefill.overhead.callback_ms += callback_ms;
            break;
        case ExecutionPhase::DECODE:
            stats_.decode.overhead.callback_ms += callback_ms;
            break;
        case ExecutionPhase::COMBINED:
            break;
        }

        PerfStatsCollector::recordTimingNs(
            "stage_executor_cpu",
            "post_graph_snapshot_callbacks",
            static_cast<uint64_t>(callback_ms * 1.0e6),
            GraphExecutorStats::currentPhase() == ExecutionPhase::DECODE ? "decode" : "prefill",
            "",
            PerfStatsCollector::Tags{
                {"attribution", "host"},
                {"source", "device_graph_executor"},
                {"context", context ? context : "post_graph"},
                {"callbacks", std::to_string(callback_count)}});

        return true;
    }

    bool DeviceGraphExecutor::publishCapturedTerminalStateAfterGraphExecution(
        ComputeGraph &graph,
        int terminal_row,
        void *producer_stream_override,
        const char *context,
        const int *device_request_seq_lens,
        int request_count,
        int request_row_width,
        const std::vector<int> *host_request_seq_lens)
    {
        if (request_count <= 0)
        {
            LOG_ERROR("[DeviceGraphExecutor] Captured terminal state publication requires a positive request count");
            return false;
        }

        /*
         * Logical shape, rather than metadata allocation, decides whether this
         * is a request-batched publication. A runner initialized with capacity
         * B may retain a stable B-entry device-length buffer while executing a
         * scalar request. That pointer is intentionally graph-stable storage;
         * its mere presence must not reinterpret an active batch of one as a
         * grouped request transaction.
        */
        const bool request_batched = request_count > 1;
        bool request_shape_validated = false;
        auto validateRequestShape = [&]() -> bool
        {
            if (request_shape_validated)
                return true;
            if (request_row_width <= 0)
            {
                LOG_ERROR("[DeviceGraphExecutor] Request-batched terminal state publication requires a positive request row width"
                          << " (request_count=" << request_count
                          << ", request_row_width=" << request_row_width << ")");
                return false;
            }
            if (request_count > std::numeric_limits<int>::max() / request_row_width)
            {
                LOG_ERROR("[DeviceGraphExecutor] Request-batched terminal state publication shape overflows "
                          "the flat row index domain"
                          << " (request_count=" << request_count
                          << ", request_row_width=" << request_row_width << ")");
                return false;
            }
            request_shape_validated = true;
            return true;
        };
        std::vector<int> host_terminal_rows;
        auto prepareHostTerminalRows = [&](const std::string &stage_name) -> bool
        {
            if (!host_request_seq_lens)
            {
                LOG_ERROR("[DeviceGraphExecutor] CPU request-batched terminal state publication"
                          << (context ? std::string(" during ") + context : std::string{})
                          << " for stage '" << stage_name
                          << "' requires host-owned request lengths");
                return false;
            }
            if (host_terminal_rows.size() == static_cast<size_t>(request_count))
                return true;
            if (host_request_seq_lens->size() < static_cast<size_t>(request_count))
            {
                LOG_ERROR("[DeviceGraphExecutor] Host request-length metadata is smaller than the active request batch"
                          << " (request_count=" << request_count
                          << ", host_length_count=" << host_request_seq_lens->size() << ")");
                return false;
            }

            host_terminal_rows.reserve(static_cast<size_t>(request_count));
            for (int request = 0; request < request_count; ++request)
            {
                const int real_length = (*host_request_seq_lens)[request];
                if (real_length <= 0 || real_length > request_row_width)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Host request length is outside the active padded row domain"
                              << " (request=" << request
                              << ", real_length=" << real_length
                              << ", request_row_width=" << request_row_width << ")");
                    return false;
                }
                host_terminal_rows.push_back(
                    request * request_row_width + real_length - 1);
            }
            return true;
        };
        auto validateRequestMetadataOwner = [&](const DeviceId &stage_device,
                                                const std::string &stage_name) -> bool
        {
            if (!validateRequestShape())
                return false;
            if (!stage_device.is_gpu())
                return prepareHostTerminalRows(stage_name);
            if (device_request_seq_lens)
                return true;

            LOG_ERROR("[DeviceGraphExecutor] GPU request-batched terminal state publication"
                      << (context ? std::string(" during ") + context : std::string{})
                      << " for stage '" << stage_name
                      << "' requires device-owned request lengths");
            return false;
        };
        if (!request_batched && terminal_row < 0)
        {
            LOG_ERROR("[DeviceGraphExecutor] Cannot publish captured terminal state"
                      << (context ? std::string(" during ") + context : std::string{})
                      << ": terminal row is negative (" << terminal_row << ")");
            return false;
        }
        const auto &order = graph.getExecutionOrder();
        if (order.empty())
            return true;

        size_t restored_count = 0;
        size_t host_restored_count = 0;
        size_t device_restored_count = 0;
        size_t skipped_count = 0;
        size_t direct_commit_count = 0;
        for (const auto &name : order)
        {
            ComputeNode *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;

            IComputeStage *stage = node->stage.get();
            DeviceId stage_device = stage->device();
            if (!stage_device.is_valid())
                stage_device = node->device;
            if (request_batched &&
                stage->requestBatchedTerminalStateCommittedDuringExecution(
                    request_count,
                    request_row_width))
            {
                if (!validateRequestMetadataOwner(stage_device, name))
                    return false;
                ++direct_commit_count;
                continue;
            }
            if (!stage->hasVerifierStateCapture())
            {
                if (stage->requiresVerifierStateCaptureForPublication())
                {
                    LOG_ERROR("[DeviceGraphExecutor] Captured terminal state publication"
                              << (context ? std::string(" during ") + context : std::string{})
                              << " required verifier-state capture for stage '"
                              << name << "' but no capture was bound");
                    return false;
                }
                ++skipped_count;
                continue;
            }

            void *producer_stream = producer_stream_override
                                        ? producer_stream_override
                                        : stage->gpuStream();
            if (stage_device.is_gpu() && !producer_stream)
            {
                LOG_ERROR("[DeviceGraphExecutor] Captured terminal state publication"
                          << (context ? std::string(" during ") + context : std::string{})
                          << " for GPU stage '" << name
                          << "' requires an explicit non-null stream");
                return false;
            }

            bool restored = false;
            if (!request_batched)
            {
                restored = stage->restoreVerifierStateCaptureRow(
                    terminal_row,
                    producer_stream);
            }
            else if (stage_device.is_gpu())
            {
                if (!validateRequestMetadataOwner(stage_device, name))
                    return false;
                restored = stage->restoreVerifierStateCaptureRequestTerminalRows(
                    device_request_seq_lens,
                    request_count,
                    request_row_width,
                    producer_stream);
                if (restored)
                    ++device_restored_count;
            }
            else
            {
                if (!validateRequestMetadataOwner(stage_device, name))
                    return false;
                restored = stage->restoreVerifierStateCaptureRows(
                    host_terminal_rows.data(),
                    request_count,
                    producer_stream);
                if (restored)
                    ++host_restored_count;
            }
            if (!restored)
            {
                LOG_ERROR("[DeviceGraphExecutor] Captured terminal state publication"
                          << (context ? std::string(" during ") + context : std::string{})
                          << (request_batched
                                  ? stage_device.is_gpu()
                                        ? " failed restoring device-owned request terminal rows"
                                        : " failed restoring host-owned request terminal rows"
                                  : " failed restoring row " + std::to_string(terminal_row))
                          << " for stage '" << name << "'");
                return false;
            }
            ++restored_count;
        }

        if (restored_count > 0)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "captured_terminal_state_publications",
                static_cast<double>(restored_count),
                GraphExecutorStats::currentPhase() == ExecutionPhase::DECODE
                    ? "decode"
                    : "prefill",
                "executor",
                {{"context", context ? context : "unknown"},
                 {"terminal_row", std::to_string(terminal_row)},
                 {"publication_policy",
                  !request_batched
                      ? "scalar_terminal_row"
                      : device_restored_count > 0 && host_restored_count > 0
                            ? "mixed_backend_request_terminal_rows"
                            : device_restored_count > 0
                                  ? "device_request_terminal_rows"
                                  : "host_request_terminal_rows"},
                 {"request_count", std::to_string(request_count)},
                 {"skipped_stages", std::to_string(skipped_count)}});
        }
        if (direct_commit_count > 0)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "request_batched_terminal_state_direct_commits",
                static_cast<double>(direct_commit_count),
                GraphExecutorStats::currentPhase() == ExecutionPhase::DECODE
                    ? "decode"
                    : "prefill",
                "executor",
                {{"context", context ? context : "unknown"},
                 {"request_count", std::to_string(request_count)},
                 {"request_row_width", std::to_string(request_row_width)},
                 {"commit_policy", "grouped_kernel_live_state_bank"}});
        }
        return true;
    }

    // executeWithGraphCapture, executeDecodeWithCapturePolicy,
    // executeWithCachedGraphReplay → DeviceGraphExecutor_GraphCapture.cpp

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

    bool DeviceGraphExecutor::validatePreparedWeightBindings(
        const ComputeNode &node,
        const StageBufferContract &contract,
        DeviceId target_device) const
    {
        for (const auto &prepared : contract.prepared_weights)
        {
            const PreparedWeightRef &ref = prepared.ref;
            auto fail = [&](const std::string &reason)
            {
                LOG_ERROR("[DeviceGraphExecutor] Invalid prepared-weight contract for stage '"
                          << node.name << "': " << reason
                          << " binding_id=" << ref.binding_id
                          << " kind=" << toString(ref.kind)
                          << " ref_device=" << ref.device.toString()
                          << " stage_device=" << target_device.toString());
                return false;
            };

            if (!prepared.source_tensor)
                return fail("missing source tensor");
            if (!prepared.store)
                return fail("missing PreparedWeightStore");
            if (ref.binding_id == 0 ||
                ref.kind == PreparedWeightKind::None ||
                !ref.device.is_valid())
            {
                return fail("incomplete PreparedWeightRef");
            }
            if (ref.device != target_device)
                return fail("PreparedWeightRef belongs to another device");
            if (!prepared.store->contains(ref))
                return fail("PreparedWeightStore does not contain the exact ref");

            const auto binding = prepared.store->binding(ref);
            if (!binding.has_value())
                return fail("PreparedWeightStore ref has no binding");
            if (binding->tensor != prepared.source_tensor)
                return fail("PreparedWeightStore binding names a different source tensor");
        }

        return true;
    }

    bool DeviceGraphExecutor::runStages(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        const StageRunPolicy &policy,
        const std::unordered_set<std::string> *collective_nodes,
        GraphSnapshotManifest *snapshot_manifest)
    {
        // Set GPU device once for the entire pass
        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

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

        if (!policy.preserve_gpu_streams &&
            !bindScheduleToWorkerStreams(schedule, ctx, config_))
        {
            return false;
        }

        // Ensure last stage is marked for event-based dirty marking
        schedule.back().node->is_final_output = true;

        // =====================================================================
        // GPU Stage Timing: event-based per-stage profiling
        // Gated by policy AND env var. ~1μs CPU overhead per event record.
        // =====================================================================
        const bool timeline_requested = PerfStatsCollector::gpuStageEventTimingEnabled();
        const bool timeline_active = policy.timeline && timeline_requested && ctx->isGPU() && !isGraphCaptureActive();
        const bool cpu_stage_timing_active =
            PerfStatsCollector::isEnabled() && ctx && !ctx->isGPU();
        IWorkerGPUContext *timeline_gpu_ctx = nullptr;
        if (timeline_active)
        {
            timeline_gpu_ctx = tryGetWorkerContext(ctx->deviceId(), config_);
            if (timeline_gpu_ctx)
            {
                stage_timeline_.ensureCapacity(timeline_gpu_ctx, schedule.size());
                stage_timeline_.resetTimings();

                // Stage metadata is graph-shape specific; refresh it for each profiled execution.
                for (size_t i = 0; i < schedule.size(); ++i)
                {
                    auto *node = schedule[i].node;
                    if (node && node->stage)
                        stage_timeline_.setStageInfo(i, node->name, node->stage->type());
                }
                stage_timeline_info_populated_ = true;
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

            if (!ensureStageGPUStreamBound(*node, ctx, config_))
            {
                notifyStageFailure(node->name, "GPU stage has no explicit stream");
                return false;
            }
            void *stage_timeline_stream = nullptr;
            if (timeline_active && timeline_gpu_ctx)
            {
                stage_timeline_stream = node->stage->gpuStream();
                if (!stage_timeline_stream)
                    stage_timeline_stream = timeline_gpu_ctx->defaultStream();
            }

            if (timeline_active && timeline_gpu_ctx)
                stage_timeline_.recordStart(i, timeline_gpu_ctx, stage_timeline_stream);

            if (debugEnv().vram_trace && ctx->isGPU())
            {
                const DeviceId stage_device = node->device.is_valid() ? node->device : node->stage->device();
                LOG_TRACE("[VRAM_TRACE] stage.before index=" << i
                                                            << " name=" << node->name
                                                            << " type=" << computeStageTypeName(node->stage->type())
                                                            << " device=" << stage_device.toString());
            }

            const auto cpu_stage_start =
                cpu_stage_timing_active ? PerfStatsCollector::Clock::now()
                                        : PerfStatsCollector::Clock::time_point{};
            bool stage_ok = false;
            try
            {
                stage_ok = runStage(
                    *node,
                    ctx,
                    policy,
                    is_coll,
                    snapshot_manifest);
            }
            catch (const std::exception &e)
            {
                const DeviceId stage_device =
                    node->device.is_valid() ? node->device : node->stage->device();
                LOG_ERROR("[DeviceGraphExecutor] Exception while running stage '"
                          << node->name
                          << "' type=" << computeStageTypeName(node->stage->type())
                          << " dynamic_type=" << typeid(*node->stage).name()
                          << " device=" << stage_device.to_string()
                          << " exception_type=" << typeid(e).name()
                          << " what=" << e.what());
                notifyStageFailure(node->name, std::string("stage threw exception: ") + e.what());
                throw;
            }
            if (!stage_ok)
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage failed: " << node->name);
                notifyStageFailure(node->name, "stage execution returned false");
                return false;
            }
            if (cpu_stage_timing_active)
            {
                const auto cpu_stage_end = PerfStatsCollector::Clock::now();
                const auto ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_stage_end - cpu_stage_start).count();
                const std::string stage_type_name = computeStageTypeName(node->stage->type());

                // Keep the coarse aggregate stable for existing dashboards, then
                // add a node-level record so verifier tuning can identify the
                // exact graph stage that dominates an M=2..4 replay.
                PerfStatsCollector::recordTimingNs(
                    "stage_cpu",
                    std::string("type.") + stage_type_name,
                    ns > 0 ? static_cast<uint64_t>(ns) : 0,
                    "execute",
                    ctx->deviceId().to_string(),
                    PerfStatsCollector::Tags{
                        {"attribution", "cpu_wall"},
                        {"source", "device_graph_executor"}});
                PerfStatsCollector::recordTimingNs(
                    "stage_cpu_detail",
                    node->name,
                    ns > 0 ? static_cast<uint64_t>(ns) : 0,
                    "execute",
                    ctx->deviceId().to_string(),
                    PerfStatsCollector::Tags{
                        {"attribution", "cpu_wall"},
                        {"source", "device_graph_executor"},
                        {"stage_type", stage_type_name}});
            }

            if (debugEnv().vram_trace && ctx->isGPU())
            {
                const DeviceId stage_device = node->device.is_valid() ? node->device : node->stage->device();
                LOG_TRACE("[VRAM_TRACE] stage.after index=" << i
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
                stage_timeline_.recordStop(i, timeline_gpu_ctx, stage_timeline_stream);
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
            LOG_TRACE("[DeviceGraphExecutor] Total execution: " << total_ms << "ms for "
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
        bool is_collective,
        GraphSnapshotManifest *snapshot_manifest)
    {
        if (!node.stage)
            return runStageImpl(
                node, ctx, policy, is_collective, snapshot_manifest);

        /*
         * Every execution path converges here.  During native capture the scope
         * advances the frozen dependency plan only after the complete canonical
         * stage runner succeeds; exceptions and false returns leave the
         * transaction failed and cannot expose a partially recorded producer.
         */
        ScopedGraphCaptureStage capture_stage(node.stage.get());
        const bool success = runStageImpl(
            node, ctx, policy, is_collective, snapshot_manifest);
        if (success)
            capture_stage.complete();
        return success;
    }

    bool DeviceGraphExecutor::runStageImpl(
        ComputeNode &node,
        IDeviceContext *ctx,
        const StageRunPolicy &policy,
        bool is_collective,
        GraphSnapshotManifest *snapshot_manifest)
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
                    LOG_DEBUG("[DeviceGraphExecutor] Attempting strided ALLGATHER intercept for '" << node.name << "'");
                    if (executeCollectiveStridedAllgather(node, ctx))
                        return true;
                    LOG_DEBUG("[DeviceGraphExecutor] Strided ALLGATHER not available, using stage execution");
                }
            }

            // LOCAL TP / GLOBAL TP: collective_ctx_ is nullptr, stage handles
            // collective internally.  In fast-decode mode (no coherence) we can
            // skip the full coherence/validation pipeline, but we must still
            // accumulate profiling stats so ALLREDUCE/ALLGATHER appear in the
            // executor timing tables.
            const StageBufferContract contract = arena_ ? node.stage->bufferContract() : StageBufferContract{};
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            const bool force_contract_coherence =
                fastPolicyRequiresContractCoherence(policy, arena_, contract, target_device);

            if (!policy.coherence && !force_contract_coherence)
            {
                if (!ensureStageGPUStreamBound(node, ctx, config_))
                    return false;

                const bool profiling_fast = policy.profiling && config_.enable_profiling;
                std::chrono::high_resolution_clock::time_point t0{};
                if (profiling_fast)
                    t0 = std::chrono::high_resolution_clock::now();

                if (debugEnv().execution.trace_stages)
                {
                    LOG_DEBUG("[StageTrace] begin stage='" << node.name
                                                          << "' type=" << static_cast<int>(node.stage->type())
                                                          << " device=" << target_device.to_string());
                }
                bool ok = node.stage->execute(ctx);
                if (debugEnv().execution.trace_stages)
                {
                    LOG_DEBUG("[StageTrace] end stage='" << node.name
                                                        << "' ok=" << (ok ? "true" : "false")
                                                        << " device=" << target_device.to_string());
                }

                if (ok && debugEnv().validation.sync_each_stage)
                {
                    DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
                    if (target_device.is_gpu())
                    {
                        if (auto *gpu_ctx = tryGetWorkerContext(target_device, config_); gpu_ctx && !gpu_ctx->debugSynchronize())
                        {
                            LOG_ERROR("[SYNC_EACH_STAGE] stage='" << node.name
                                                                  << "' device=" << target_device.to_string()
                                                                  << " device debug synchronization failed");
                            ok = false;
                        }
                    }
                }

                /*
                 * Fast collective execution is still a complete stage
                 * boundary, even though this branch returns before the normal
                 * snapshot section at the bottom of runStage().  Therefore it
                 * must reproduce both halves of that section's GPU contract:
                 *
                 * 1. Record a D2D copy immediately after the collective.  When
                 *    stream capture is active this becomes a graph node and
                 *    preserves the exact post-allreduce bytes before a later
                 *    arena alias can overwrite the source buffer.
                 * 2. Publish the copied slot immediately only for eager
                 *    execution.  Captured execution cannot perform D2H work or
                 *    invoke host callbacks inside the graph; its caller
                 *    publishes the immutable slot manifest after launch.
                 *
                 * policy.snapshot_callback controls immediate host
                 * publication, not whether the graph-stable D2D copy exists.
                 * Conditioning the copy on !policy.snapshot_callback left
                 * *_ALLREDUCED diagnostics permanently stuck at warmup bytes
                 * for monolithic prefill graphs.
                 */
                if (ok && config_.snapshot_callback)
                {
                    DeviceId snapshot_device = target_device;
                    if (!snapshot_device.is_valid() && ctx)
                        snapshot_device = ctx->deviceId();
                    GraphSnapshotManifest &manifest =
                        snapshot_manifest ? *snapshot_manifest
                                          : transient_snapshot_manifest_;
                    if (!captureGraphSnapshotCopies(
                            node,
                            snapshot_device,
                            node.stage->gpuStream(),
                            manifest))
                    {
                        ok = false;
                    }
                    else if (policy.snapshot_callback &&
                             !isGraphCaptureActive() &&
                             !publishGraphSnapshotCopies(
                                 node.name,
                                 node.stage->gpuStream(),
                                 manifest))
                    {
                        ok = false;
                    }
                }

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

                    const bool stage_is_collective =
                        node.stage->isCollectiveStage();
                    if (stage_is_collective)
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
                        if (stage_is_collective)
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
        // Stage Coherence: arena contract input/output + weight uploads
        // =====================================================================
        // Contract is ALWAYS fetched when we have an arena. mark_dirty (which
        // uses the contract) is unconditional because marking outputs
        // DEVICE_AUTHORITATIVE is a correctness requirement, not overhead.
        // The coherence flag controls whether prepareForRead/Write are called.
        const StageBufferContract contract = arena_ ? node.stage->bufferContract() : StageBufferContract{};
        const bool use_contract = !contract.empty() && arena_ != nullptr;
        DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
        const bool force_contract_coherence =
            use_contract && fastPolicyRequiresContractCoherence(policy, arena_, contract, target_device);

        // Bind GPU stream early so coherence operations (H2D/D2H) run on
        // the same stream as the stage's compute kernels. Debug paths that
        // materialize GPU outputs also require this explicit stream.
        if (!ensureStageGPUStreamBound(node, ctx, config_))
            return false;
        void *stage_stream = node.stage ? node.stage->gpuStream() : nullptr;

        // Prepared representations are a graph-construction invariant, not a
        // coherence side effect. Validate them before any raw source tensor can
        // be uploaded or observed by a stage.
        std::string prepared_error;
        if (!node.stage->validatePreparedWeights(&prepared_error))
        {
            LOG_ERROR("[DeviceGraphExecutor] Prepared weight validation failed for stage '"
                      << node.name << "': " << prepared_error);
            return false;
        }
        if (!validatePreparedWeightBindings(node, contract, target_device))
            return false;

        // =====================================================================
        // Stage Dump: input snapshot setup
        // =====================================================================
        StageDumpContext dump_ctx;
        const bool should_dump = policy.stage_dump && StageDumper::shouldDump(
                                                          node.stage.get(),
                                                          node.name,
                                                          config_.current_layer_idx,
                                                          config_.current_iteration,
                                                          config_.mpi_rank);

        // StageBufferContract is the coherence source of truth. Pull dump info
        // before execute only for consumers that actually inspect it before the
        // stage runs.
        const bool need_pre_execute_dump_info =
            should_dump ||
            (policy.pointer_validation &&
             debugEnv().validation.validate_gpu_ptrs &&
             target_device.is_gpu());
        StageDumpInfo cached_dump_info{};
        if (need_pre_execute_dump_info)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            cached_dump_info = node.stage->getDumpInfoSnapshot();
            if (profiling)
            {
                phase_end = std::chrono::high_resolution_clock::now();
                get_dump_info_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
            }
        }

        if (policy.coherence || force_contract_coherence)
        {
            auto coh_policy = node.stage->coherencePolicy();

            LOG_TRACE("[DeviceGraphExecutor] Stage '" << node.name << "' coherencePolicy=" << toString(coh_policy)
                                                      << " target_device=" << target_device.to_string()
                                                      << " use_contract=" << use_contract
                                                      << " forced_contract=" << force_contract_coherence);

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

                // Weight residency is not arena-managed. GPU preparation owns
                // placement before graph execution; the executor may only
                // validate exact residency and join an existing producer event
                // to this stage stream before exposing a raw weight pointer.
                // A missing allocation is fatal here and can never trigger a
                // lazy upload in the inference hot path.
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
                            if (target_device.is_gpu())
                            {
                                TransferEngine::requireDeviceInput(
                                    tb, target_device, stage_stream);
                            }
                            else
                            {
                                TransferEngine::prepareHostInput(tb);
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

                for (const auto &binding : contract.writesRequiringPrepare())
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
#if LLAMINAR_ASSERTIONS_ACTIVE
        if (policy.validation && debugEnv().validation.validate_inputs)
        {
            verifyStageEntry(node, layer_idx);
        }
#endif

        // =====================================================================
        // Stage Dump: input snapshots
        // =====================================================================
        if (should_dump)
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage dump requested during GPU graph capture for stage '"
                          << node.name
                          << "'. Stage dumps materialize tensor payloads on host and are not graph-capturable.");
                return false;
            }

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
                auto *gpu_ctx = tryGetWorkerContext(target_device, config_);
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
        if (debugEnv().execution.trace_stages)
        {
            LOG_DEBUG("[StageTrace] begin stage='" << node.name
                                                  << "' type=" << static_cast<int>(node.stage->type())
                                                  << " device=" << target_device.to_string());
        }
        bool success = node.stage->execute(ctx);
        if (debugEnv().execution.trace_stages)
        {
            LOG_DEBUG("[StageTrace] end stage='" << node.name
                                                << "' ok=" << (success ? "true" : "false")
                                                << " device=" << target_device.to_string());
        }

        if (success && debugEnv().validation.sync_each_stage)
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            if (target_device.is_gpu())
            {
                if (auto *gpu_ctx = tryGetWorkerContext(target_device, config_); gpu_ctx && !gpu_ctx->debugSynchronize())
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
                                    || (policy.validation && debugEnv().validation.validate_buffers)
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

            const bool needs_post_execute_debug_info =
                debugEnv().validation.trace_local_tp_pointer ||
                debugEnv().stage_output_print.shouldPrint(node.name);
            if (needs_post_execute_debug_info)
            {
                StageDumpInfo post_execute_dump_info = node.stage->refreshDumpInfoSnapshot();
                logWatchedPointerProducer(
                    node.name,
                    post_execute_dump_info,
                    tryGetWorkerContext(
                        node.device.is_valid() ? node.device : node.stage->device(),
                        config_));
                printStageOutputs(node.name, post_execute_dump_info, node.stage->gpuStream());
            }
        }

        if (success && config_.snapshot_callback && !policy.snapshot_callback)
        {
            DeviceId snapshot_device = target_device;
            if (!snapshot_device.is_valid() && ctx)
                snapshot_device = ctx->deviceId();
            GraphSnapshotManifest &manifest =
                snapshot_manifest ? *snapshot_manifest
                                  : transient_snapshot_manifest_;
            if (!captureGraphSnapshotCopies(
                    node,
                    snapshot_device,
                    node.stage->gpuStream(),
                    manifest))
                success = false;
        }

        if (success && stageChecksumTraceEnabled() && stageChecksumTraceMatches(node.name))
        {
            StageDumpInfo checksum_dump_info = node.stage->refreshDumpInfoSnapshot();
            traceStageOutputChecksums(node.name, node.stage.get(), checksum_dump_info, node.stage->gpuStream());
        }

        // =====================================================================
        // Stage Dump: output snapshots
        // =====================================================================
        if (should_dump && success)
        {
            const auto &dump_cfg = debugEnv().stage_dump;
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();

            if (dump_cfg.async_dump)
            {
                // Rebuild post-execute dump info before output dumping. Some
                // stages populate diagnostic outputs during execute(), and the
                // async dumper consumes the StageDumpInfo passed here.
                StageDumpInfo output_dump_info = node.stage->refreshDumpInfoSnapshot();
                AsyncStageDumper::enqueueOutputs(dump_ctx, output_dump_info, node.stage->gpuStream());
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
        if (success &&
            policy.snapshot_callback &&
            config_.snapshot_callback)
        {
            if (profiling)
                phase_start = std::chrono::high_resolution_clock::now();
            DeviceId snapshot_device = node.device.is_valid() ? node.device : node.stage->device();
            if (!snapshot_device.is_valid() && ctx)
                snapshot_device = ctx->deviceId();
            /*
             * GPU graph capture has a two-phase snapshot contract:
             *
             * 1. While capture is active, record only device-to-device copies
             *    from the stage output into graph-stable scratch. Host
             *    materialization is forbidden here because ROCm invalidates the
             *    stream capture when a D2H transfer or host synchronization is
             *    inserted into the captured body.
             * 2. After capture/replay launches the graph,
             *    publishSnapshotsAfterGraphExecution() publishes directly from
             *    the immutable captured-slot manifest.
             *
             * Eager GPU execution uses the same slot manifest and publishes it
             * immediately. Only CPU execution retains live StageDumpInfo
             * publication because CPU stages have no graph-captured D2D slots.
             */
            const bool graph_capture_active = isGraphCaptureActive();
            if (snapshot_device.is_gpu())
            {
                GraphSnapshotManifest &manifest =
                    snapshot_manifest ? *snapshot_manifest
                                      : transient_snapshot_manifest_;
                if (!captureGraphSnapshotCopies(
                        node,
                        snapshot_device,
                        node.stage->gpuStream(),
                        manifest))
                {
                    success = false;
                }
                else if (!graph_capture_active &&
                         !publishGraphSnapshotCopies(
                             node.name,
                             node.stage->gpuStream(),
                             manifest))
                {
                    success = false;
                }
            }
            else
            {
                StageDumpInfo snapshot_dump_info =
                    node.stage->refreshDumpInfoSnapshot();
                if (shouldCaptureSnapshotStage(
                        node.name, snapshot_dump_info))
                {
                    snapshot_dump_info.ensureOutputsOnHost(
                        node.stage->gpuStream());
                    LOG_DEBUG(
                        "[DeviceGraphExecutor::runStage] Invoking callback for "
                        << node.name);
                    config_.snapshot_callback(node.name, snapshot_dump_info);
                }
            }
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

            const bool stage_is_collective =
                node.stage->isCollectiveStage();
            if (stage_is_collective)
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
                if (stage_is_collective)
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

    static bool stageChecksumTraceEnabled()
    {
        return debugEnv().runtime_debug.stage_checksum_trace;
    }

    static bool stageChecksumTraceMatches(const std::string &stage_name)
    {
        const auto &filter = debugEnv().runtime_debug.stage_checksum_filter;
        return filter.empty() || stage_name.find(filter) != std::string::npos;
    }

    static uint64_t fnv1a64(const uint8_t *data, size_t size)
    {
        uint64_t hash = 1469598103934665603ULL;
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= static_cast<uint64_t>(data[i]);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    static void traceStageOutputChecksums(
        const std::string &stage_name,
        const IComputeStage *stage,
        const StageDumpInfo &dump_info,
        void *stream)
    {
        if (!stageChecksumTraceEnabled() || !stageChecksumTraceMatches(stage_name))
            return;

        if (dump_info.outputs.empty())
        {
            LOG_DEBUG("[StageChecksum] stage='" << stage_name
                                               << "' type=" << (stage ? computeStageTypeName(stage->type()) : "(unknown)")
                                               << " outputs=0");
            return;
        }

        dump_info.ensureOutputsOnHost(stream);
        for (const auto &output : dump_info.outputs)
        {
            if (!output.data)
            {
                LOG_DEBUG("[StageChecksum] stage='" << stage_name
                                                   << "' output='" << (output.name ? output.name : "(unnamed)")
                                                   << "' data=null");
                continue;
            }

            const size_t logical_elements = output.rows * output.cols;
            const size_t byte_size = output.byte_size ? output.byte_size : logical_elements * output.element_size;
            const auto *bytes = static_cast<const uint8_t *>(output.data);
            const uint64_t hash = fnv1a64(bytes, byte_size);

            std::ostringstream oss;
            oss << "[StageChecksum] stage='" << stage_name
                << "' type=" << (stage ? computeStageTypeName(stage->type()) : "(unknown)")
                << " output='" << (output.name ? output.name : "(unnamed)")
                << "' shape=[" << output.rows << "x" << output.cols << "]"
                << " dtype=" << (output.dtype ? output.dtype : "(unknown)")
                << " bytes=" << byte_size
                << " hash=0x" << std::hex << hash << std::dec;

            if (output.dtype && std::strcmp(output.dtype, "FP32") == 0 && output.element_size == sizeof(float))
            {
                const auto *fp = static_cast<const float *>(output.data);
                double sum = 0.0;
                double abs_sum = 0.0;
                double max_abs = 0.0;
                for (size_t i = 0; i < logical_elements; ++i)
                {
                    const double v = static_cast<double>(fp[i]);
                    sum += v;
                    const double av = std::fabs(v);
                    abs_sum += av;
                    max_abs = std::max(max_abs, av);
                }
                oss << " sum=" << std::setprecision(12) << sum
                    << " abs_sum=" << abs_sum
                    << " max_abs=" << max_abs;
                const size_t sample_count = std::min<size_t>(logical_elements, 4);
                oss << " first=[";
                for (size_t i = 0; i < sample_count; ++i)
                {
                    if (i)
                        oss << ",";
                    oss << fp[i];
                }
                oss << "]";
            }

            LOG_DEBUG(oss.str());
        }
    }

    /**
     * @brief Print first N elements of stage outputs for debugging
     *
     * Called after TransferEngine publishes the stage's exact producer event.
     * Controlled by LLAMINAR_STAGE_OUTPUT_PRINT environment variable.
     */
    static void printStageOutputs(const std::string &stage_name, const StageDumpInfo &dump_info, void *stream)
    {
        const auto &config = debugEnv().stage_output_print;
        if (!config.shouldPrint(stage_name))
        {
            return;
        }

        const int num_elements = config.num_elements;
        const int num_rows = config.num_rows;

        dump_info.ensureOutputsOnHost(stream);

        for (const auto &output : dump_info.outputs)
        {
            if (!output.tensor || !output.data)
            {
                continue;
            }

            const float *data = static_cast<const float *>(output.data);

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
        const bool is_collective =
            node.stage && node.stage->isCollectiveStage();
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
        StageDumpInfo dump_info = stage->getDumpInfoSnapshot();
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
        StageDumpInfo dump_info = stage->getDumpInfoSnapshot();

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
