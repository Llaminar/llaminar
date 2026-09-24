/**
 * @file BenchmarkMode.cpp
 * @brief Release benchmark entry point sharing production readiness and policy.
 *
 * Request termination belongs to this mode; process finalization belongs to
 * the caller's MPIProcessSession. Returning first retires mode-local adapters
 * and handlers before the runner, contexts and outer session are destroyed.
 *
 * The orchestration runner owns model preparation and topology-resolved runtime
 * defaults. This mode drives the requested workload and exports that resolved
 * configuration without adding benchmark-only calibration or device policy.
 */

#include "app/modes/BenchmarkMode.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "utils/KernelProfiler.h"
#include "utils/BenchmarkRunner.h"

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

        /**
         * @brief Report the already-published production prefill policy.
         *
         * Benchmarking must not change graph or bucket settings according to
         * topology. The same 512-token request works through the serving
         * runner's captured chunk scheduler even when a PP graph has a
         * 256-row activation arena. Disabling buckets here used to bypass that
         * scheduler and send the whole prompt to one undersized graph.
         */
        void logBenchmarkPrefillPolicy()
        {
            const auto &env = debugEnv();
            const bool user_selected_bucket_mode = env.presence.has("LLAMINAR_PREFILL_GRAPH_BUCKETS");
            const bool user_selected_bucket_sizes = env.presence.has("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES");
            const bool user_selected_gpu_graphs = env.presence.has("LLAMINAR_GPU_GRAPHS");
            const auto &exec = env.execution;
            LOG_INFO("[Benchmark] Prefill graph buckets "
                     << (exec.prefill_graph_buckets ? "enabled" : "disabled")
                     << "; bucket_sizes=" << formatBucketList(exec.prefill_graph_bucket_sizes)
                     << (user_selected_bucket_sizes ? " (bucket env override)" : " (production default)")
                     << "; mode=" << (user_selected_bucket_mode ? "env override" : "production default")
                     << "; gpu_graphs=" << (exec.gpu_graphs ? "enabled" : "disabled")
                     << (user_selected_gpu_graphs ? " (env override)" : ""));
        }

        /**
         * @brief Terminate the request channel while retaining outer MPI ownership.
         * @param ctx Active caller-owned application resources.
         * @param detail Failure shared with followers before runner retirement.
         * @return Nonzero command exit; the caller's scope finalizes MPI later.
         */
        int shutdownAfterUnhandledException(AppContext &ctx, const std::string &detail)
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
            return 0;
        }

        if (mpi_coordinated)
            runner->setMPICoordinatedMode(true);

        auto shutdownAndReturn = [&](bool success, const std::string &failure_reason = {}) -> int
        {
            MoEExpertOverlayProfiler::flush();
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            return success ? 0 : 1;
        };

        LOG_DEBUG("Running benchmark mode on coordinated authority rank "
                  << runner->coordinatedRootRank() << "...");

        logBenchmarkPrefillPolicy();

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
            return shutdownAndReturn(
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
                return shutdownAndReturn(false, "failed to write benchmark JSON");
            }
            // The runner exposes the topology-resolved startup configuration;
            // raw CLI intent cannot attest which hardware defaults ran.
            json_out << benchmarkResultToJsonString(result, &runner->config()) << '\n';
            if (!json_out)
            {
                LOG_ERROR("Failed to write benchmark JSON output path: "
                          << ctx.config.benchmark_json_output_path);
                MoEExpertOverlayProfiler::flush();
                return shutdownAndReturn(false, "failed to write benchmark JSON");
            }
            LOG_INFO("Benchmark JSON written to " << ctx.config.benchmark_json_output_path);
        }
        MoEExpertOverlayProfiler::flush();

        return shutdownAndReturn(
            result.success,
            result.success ? std::string{} : "benchmark runner reported failure");
    }
    catch (const std::exception &e)
    {
        return shutdownAfterUnhandledException(ctx, e.what());
    }
    catch (...)
    {
        return shutdownAfterUnhandledException(ctx, "unknown exception");
    }

} // namespace llaminar2
