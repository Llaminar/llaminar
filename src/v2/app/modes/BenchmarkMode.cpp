/**
 * @file BenchmarkMode.cpp
 * @brief Benchmark mode (--benchmark)
 */

#include "app/modes/BenchmarkMode.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "execution/moe/MoERebalanceController.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/BenchmarkRunner.h"
#include "app/MPIShutdown.h"

#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /// @brief Format prefill bucket sizes for the benchmark startup log.
        std::string formatBucketList(const std::vector<int> &buckets)
        {
            std::ostringstream oss;
            for (size_t i = 0; i < buckets.size(); ++i)
            {
                if (i > 0)
                    oss << ",";
                oss << buckets[i];
            }
            return oss.str();
        }

        /// @brief Detect whether this benchmark run executes a multi-device
        ///        (tensor- or pipeline-parallel) configuration whose per-device
        ///        graphs contain collective stages.
        ///
        /// Padded bucketed prefill is intentionally unsupported when collective
        /// nodes are present (Phase 6 fail-loud guard in ForwardExecutionEngine):
        /// padding the sequence to a bucket length would desynchronize collective
        /// sizes across participants. For these runs we must execute the exact
        /// prefill length instead of a padded bucket.
        ///
        /// @param config  Parsed orchestration configuration for this run.
        /// @param mpi_ctx MPI context (used to detect global/multi-rank TP).
        /// @return true if the run uses TP/PP collectives and must not bucket prefill.
        bool benchmarkUsesCollectives(const OrchestrationConfig &config,
                                      const std::shared_ptr<IMPIContext> &mpi_ctx)
        {
            // Simple tensor parallelism (degree or explicit device list).
            if (config.tp_degree > 1 || config.tp_devices.size() > 1)
                return true;
            // Hybrid local/global TP degrees.
            if (config.tp_local_degree > 1 || config.tp_global_degree > 1)
                return true;
            // Pipeline parallelism (simple degree or named domains / pp-stages).
            if (config.pp_degree > 1 || config.usesNamedDomains())
                return true;
            // Global TP/PP distributed across multiple MPI ranks.
            if (mpi_ctx && mpi_ctx->world_size() > 1)
                return true;
            return false;
        }

        /// @brief Opt benchmark mode into production bucketed prefill defaults.
        void configureBenchmarkPrefillBuckets(const std::shared_ptr<IMPIContext> &mpi_ctx,
                                               const OrchestrationConfig &config)
        {
            const auto &env = debugEnv();
            const bool user_selected_bucket_mode = env.presence.has("LLAMINAR_PREFILL_GRAPH_BUCKETS");
            const bool user_selected_bucket_sizes = env.presence.has("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES");
            const bool user_selected_gpu_graphs = env.presence.has("LLAMINAR_GPU_GRAPHS");

            // Multi-device TP/PP runs cannot use padded bucketed prefill (collective
            // stages would desynchronize). Only auto-enable bucketing for
            // single-device runs; an explicit user opt-in is still honored (and will
            // hit the fail-loud guard if it is incompatible with collectives).
            const bool uses_collectives = benchmarkUsesCollectives(config, mpi_ctx);

            if (!user_selected_bucket_mode)
            {
                if (uses_collectives)
                {
                    if (mpi_ctx && mpi_ctx->rank() == 0)
                    {
                        LOG_INFO("[Benchmark] Multi-device (TP/PP) run detected; leaving prefill "
                                 "graph bucketing disabled (padded buckets are incompatible with "
                                 "collective stages — running exact prefill length)");
                    }
                }
                else
                {
                    setenv("LLAMINAR_PREFILL_GRAPH_BUCKETS", "1", 1);
                }
            }
            else if (!env.execution.prefill_graph_buckets && !user_selected_gpu_graphs)
            {
                setenv("LLAMINAR_GPU_GRAPHS", "0", 1);
            }

            // Leave LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES unset unless the user
            // explicitly supplied it. DebugEnv then reloads the production
            // geometric bucket ladder instead of a benchmark-prompt-sized list.
            auto &mutable_env = mutableDebugEnv();
            mutable_env.presence.reload();
            mutable_env.execution.reload();

            if (mpi_ctx && mpi_ctx->rank() == 0)
            {
                const auto &exec = debugEnv().execution;
                LOG_INFO("[Benchmark] Prefill graph buckets "
                         << (exec.prefill_graph_buckets ? "enabled" : "disabled")
                         << "; bucket_sizes=" << formatBucketList(exec.prefill_graph_bucket_sizes)
                         << (user_selected_bucket_sizes ? " (bucket env override)" : " (production default)")
                         << "; mode=" << (user_selected_bucket_mode ? "env override" : "benchmark default")
                         << "; gpu_graphs=" << (exec.gpu_graphs ? "enabled" : "disabled")
                         << (user_selected_gpu_graphs ? " (env override)" : ""));
            }
        }

        int finalizeAfterUnhandledException(AppContext &ctx, const std::string &detail)
        {
            const bool has_mpi = ctx.mpi_ctx != nullptr;
            const bool is_root = !has_mpi || ctx.mpi_ctx->rank() == 0;
            const bool notify_workers = has_mpi && ctx.mpi_ctx->world_size() > 1 && ctx.mpi_ctx->rank() == 0;

            if (is_root)
                LOG_ERROR("Benchmark mode failed with unhandled exception: " << detail);

            if (ctx.runner)
            {
                if (notify_workers)
                    ctx.runner->abortMPIWorkers(detail);
                ctx.runner->shutdown();
            }

            MoEExpertOverlayProfiler::flush();
            mpiShutdown();
            return 1;
        }

    } // namespace

    bool BenchmarkMode::matches(const OrchestrationConfig &config) const
    {
        return config.benchmark_mode;
    }

    int BenchmarkMode::execute(AppContext &ctx)
    try
    {
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        auto shutdownAndFinalize = [&](bool success, const std::string &failure_reason = {}) -> int
        {
            MoEExpertOverlayProfiler::flush();
            runner->shutdown();
            mpiShutdown();
            return success ? 0 : 1;
        };

        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("Running benchmark mode...");
        }

        configureBenchmarkPrefillBuckets(mpi_ctx, ctx.config);

        auto adapter = std::make_shared<InferenceRunnerAdapter>(runner.get());

        BenchmarkRunner benchmark(adapter, tokenizer, mpi_ctx);

        // Set up MoE expert rebalancing (incremental, during decode)
        if (auto *orch_runner = dynamic_cast<OrchestrationRunner *>(runner.get()))
        {
            if (auto *controller = orch_runner->moeRebalanceController())
            {
                if (controller->mode() == MoERebalanceMode::DYNAMIC)
                {
                    // Strategy: Do one swap-based rebalance after warmup using
                    // the 128-token histogram, then run benchmark with zero
                    // ongoing overhead. Per-step callback only tracks histogram
                    // (no rebalancing) for profiling summary.
                    benchmark.setPostWarmupCallback([orch_runner, controller, &mpi_ctx]()
                                                    {
                        orch_runner->applyMoERebalanceWithReplicas(/*log_histogram_summary=*/true);
                        if (mpi_ctx->rank() == 0)
                        {
                    LOG_DEBUG("[MoE] Post-warmup setup complete"
                              << (controller->hasReplicas()
                                  ? " (with per-token replica dispatch)"
                                  : " (local rebalance only)"));
                        } });
                    // No per-step rebalancing — the post-warmup placement
                    // is used for all benchmark iterations.
                }
            }
        }

        BenchmarkResult result = benchmark.run(ctx.config);
        benchmark.printResults(result);
        MoEExpertOverlayProfiler::flush();

        // Log MoE histogram if controller is active
        if (auto *orch_runner = dynamic_cast<OrchestrationRunner *>(runner.get()))
        {
            if (auto *controller = orch_runner->moeRebalanceController())
            {
                controller->logHistogramSummary();

                // Print MoE profiling summary when LLAMINAR_PROFILING=1
                if (KernelProfiler::isEnabled())
                {
                    std::print("{}", controller->getProfilingSummary());
                }
            }
        }

        return shutdownAndFinalize(
            result.success,
            result.success ? std::string{} : "benchmark runner reported failure");
    }
    catch (const std::exception &e)
    {
        return finalizeAfterUnhandledException(ctx, e.what());
    }
    catch (...)
    {
        return finalizeAfterUnhandledException(ctx, "unknown exception");
    }

} // namespace llaminar2
