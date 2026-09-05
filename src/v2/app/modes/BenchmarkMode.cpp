/**
 * @file BenchmarkMode.cpp
 * @brief Benchmark mode (--benchmark)
 */

#include "app/modes/BenchmarkMode.h"
#include "app/modes/BenchmarkPrefillBucketPolicy.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/BenchmarkRunner.h"
#include "app/MPIShutdown.h"

#include <cstdlib>
#include <fstream>
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

            /*
             * ExpertOverlay owns an explicit heterogeneous segmented-capture
             * protocol. Its root-published physical bucket is common to every
             * rank and participant, including a padded final segment, so its
             * collective shapes cannot diverge. Ordinary TP/PP configurations
             * still need the conservative fail-loud policy below.
             */
            const bool segmented_collective_capture_authority =
                config.moe_routed_expert_plan &&
                config.moe_routed_expert_plan
                    ->usesExpertOverlayAuthority();
            const BenchmarkPrefillBucketDisableReason disable_reason =
                benchmarkPrefillBucketDisableReason(
                    uses_collectives,
                    /*moe_rebalancing_active=*/false,
                    segmented_collective_capture_authority);

            if (!user_selected_bucket_mode)
            {
                if (disable_reason != BenchmarkPrefillBucketDisableReason::None)
                {
                    setenv("LLAMINAR_PREFILL_GRAPH_BUCKETS", "0", 1);
                    const char *reason = benchmarkPrefillBucketDisableMessage(disable_reason);
                    LOG_INFO("[Benchmark] " << reason
                                            << " — leaving prefill graph bucketing disabled "
                                               "(running exact prefill length)");
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

            const auto &exec = debugEnv().execution;
            LOG_INFO("[Benchmark] Prefill graph buckets "
                     << (exec.prefill_graph_buckets ? "enabled" : "disabled")
                     << "; bucket_sizes=" << formatBucketList(exec.prefill_graph_bucket_sizes)
                     << (user_selected_bucket_sizes ? " (bucket env override)" : " (production default)")
                     << "; mode=" << (user_selected_bucket_mode ? "env override" : "benchmark default")
                     << "; gpu_graphs=" << (exec.gpu_graphs ? "enabled" : "disabled")
                     << (user_selected_gpu_graphs ? " (env override)" : ""));
        }

        int finalizeAfterUnhandledException(AppContext &ctx, const std::string &detail)
        {
            const bool has_mpi = ctx.mpi_ctx != nullptr;
            const bool is_authority =
                ctx.runner &&
                ctx.coordinatedRequestRole() ==
                    CoordinatedRequestRole::Authority;
            const bool notify_workers =
                has_mpi && ctx.mpi_ctx->world_size() > 1 && is_authority;

            if (is_authority)
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

        const bool mpi_coordinated = mpi_ctx->world_size() > 1;
        const CoordinatedRequestRole request_role =
            ctx.coordinatedRequestRole();
        const bool is_authority =
            request_role == CoordinatedRequestRole::Authority;
        if (mpi_coordinated && !is_authority)
        {
            /*
             * BenchmarkRunner is the request controller, not a rank-local
             * graph driver.  Exactly one controller must issue clear/prefill/
             * decode commands; every other rank remains in the production
             * command loop and participates when the continuation authority
             * admits a command.
             * Running one BenchmarkRunner per rank creates two competing MPI
             * collective schedules (for example CLEAR_CACHE versus PREFILL).
             */
            LOG_DEBUG("Rank " << mpi_ctx->rank()
                              << " entering MPI worker loop for benchmark inference");
            runner->setMPICoordinatedMode(true);
            runner->runMPIWorkerLoop();
            runner->shutdown();
            MoEExpertOverlayProfiler::flush();
            mpiShutdown();
            return 0;
        }

        if (mpi_coordinated)
            runner->setMPICoordinatedMode(true);

        auto shutdownAndFinalize = [&](bool success, const std::string &failure_reason = {}) -> int
        {
            MoEExpertOverlayProfiler::flush();
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            mpiShutdown();
            return success ? 0 : 1;
        };

        LOG_DEBUG("Running benchmark mode on coordinated authority rank "
                  << runner->coordinatedRootRank() << "...");

        configureBenchmarkPrefillBuckets(mpi_ctx, ctx.config);

        /*
         * Benchmarking and serving share this exact application-startup
         * boundary. The orchestration layer owns any production-shaped setup;
         * BenchmarkRunner receives an already-ready model and times only the
         * caller-requested workload.
         */
        if (!runner->prepareForInference())
        {
            LOG_ERROR(
                "Inference runtime did not become ready for benchmarking: "
                << runner->lastError());
            return shutdownAndFinalize(
                false, "inference runtime preparation failed");
        }

        auto adapter = std::make_shared<InferenceRunnerAdapter>(runner.get());

        BenchmarkRunner benchmark(adapter, tokenizer);

        BenchmarkResult result = benchmark.run(ctx.config);
        benchmark.printResults(result);
        if (!ctx.config.benchmark_json_output_path.empty())
        {
            std::ofstream json_out(ctx.config.benchmark_json_output_path);
            if (!json_out)
            {
                LOG_ERROR("Failed to open benchmark JSON output path: "
                          << ctx.config.benchmark_json_output_path);
                MoEExpertOverlayProfiler::flush();
                return shutdownAndFinalize(false, "failed to write benchmark JSON");
            }
            json_out << benchmarkResultToJsonString(result, &ctx.config) << '\n';
            if (!json_out)
            {
                LOG_ERROR("Failed to write benchmark JSON output path: "
                          << ctx.config.benchmark_json_output_path);
                MoEExpertOverlayProfiler::flush();
                return shutdownAndFinalize(false, "failed to write benchmark JSON");
            }
            LOG_INFO("Benchmark JSON written to " << ctx.config.benchmark_json_output_path);
        }
        MoEExpertOverlayProfiler::flush();

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
