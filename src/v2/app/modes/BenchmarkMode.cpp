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

#include <memory>
#include <string>

namespace llaminar2
{
    namespace
    {
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
            LOG_INFO("Running benchmark mode...");
        }

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
                            LOG_INFO("[MoE] Post-warmup setup complete"
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
