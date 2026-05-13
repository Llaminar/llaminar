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
        if (auto* orch_runner = dynamic_cast<OrchestrationRunner*>(runner.get()))
        {
            if (auto* controller = orch_runner->moeRebalanceController())
            {
                if (controller->mode() == MoERebalanceMode::DYNAMIC)
                {
                    // Strategy: Do one swap-based rebalance after warmup using
                    // the 128-token histogram, then run benchmark with zero
                    // ongoing overhead. Per-step callback only tracks histogram
                    // (no rebalancing) for profiling summary.
                    benchmark.setPostWarmupCallback([orch_runner, controller, &mpi_ctx]() {
                        controller->logHistogramSummary();

                        std::vector<std::vector<std::vector<bool>>> gpu_cache_masks;
                        const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
                        const bool gpu_cache_enabled = gpu_cache_experts > 0;
                        if (gpu_cache_enabled)
                            gpu_cache_masks = controller->computeGpuCacheExpertMasks(gpu_cache_experts);

                        // Propose replicas BEFORE rebalance (which resets the histogram window)
                        int max_replicas = gpu_cache_enabled ? 0 : debugEnv().moe_rebalance.max_replicas;
                        if (max_replicas > 0)
                        {
                            auto replicas = controller->proposeReplicas(max_replicas);
                            if (replicas.num_replicated > 0 && mpi_ctx->rank() == 0)
                            {
                                LOG_INFO("[MoE] Expert replication: " << replicas.num_replicated
                                         << " experts replicated for per-token dispatch");
                            }
                        }

                        auto old_placement = controller->currentPlacement();
                        auto new_placement = controller->rebalance();

                        // Sync replica owner_socket to post-rebalance placement.
                        // Without this, assignForToken() uses stale ownership
                        // and dispatches experts to sockets without GEMM engines.
                        controller->syncReplicaPlacement();

                        if (new_placement.empty())
                        {
                            if (mpi_ctx->rank() == 0)
                                LOG_INFO("[MoE] Post-warmup: no beneficial swaps found");
                        }
                        else
                        {
                            int moved = 0;
                            for (int e = 0; e < static_cast<int>(old_placement.size()); ++e)
                                if (old_placement[e] != new_placement[e]) moved++;

                            if (mpi_ctx->rank() == 0)
                                LOG_INFO("[MoE] Post-warmup rebalance: "
                                         << moved << " experts moved");
                        }

                        // Transfer pre-packed weights for replicas via MPI
                        // (avoids VNNI repacking from raw on the non-owner socket)
                        ReceivedWeightsMap received;
                        if (controller->hasReplicas())
                        {
                            received = orch_runner->transferReplicaWeights(
                                controller->currentReplicas(),
                                controller->numLayers());
                        }

                        int socket_id = mpi_ctx->rank();

                        // Apply masks (includes replicas if active)
                        if (!gpu_cache_masks.empty())
                        {
                            if (!orch_runner->applyMoEExpertMasksForAllLocalDevices(gpu_cache_masks))
                            {
                                if (socket_id >= 0 && socket_id < static_cast<int>(gpu_cache_masks.size()))
                                    orch_runner->applyMoEExpertMasks(gpu_cache_masks[socket_id], received);
                            }
                        }
                        else if (!orch_runner->applyMoEExpertMasksForAllLocalDevices(*controller))
                        {
                            auto masks = controller->computeExpertMasks(socket_id);
                            orch_runner->applyMoEExpertMasks(masks, received);
                        }

                        // Set replica dispatch info on stages
                        if (controller->hasReplicas())
                        {
                            orch_runner->setExpertReplicaSet(
                                controller->currentReplicas(), socket_id);
                        }

                        // Release raw expert weight data (env: LLAMINAR_MOE_RELEASE_RAW_WEIGHTS=1)
                        // Safe because all engines are prepared and prepacked transfer is available.
                        if (debugEnv().moe_rebalance.release_raw_weights)
                        {
                            size_t freed = orch_runner->releaseRawExpertWeights();
                            if (mpi_ctx->rank() == 0)
                                LOG_INFO("[MoE] Released " << (freed >> 20) << " MB raw expert weights");
                        }

                        if (mpi_ctx->rank() == 0)
                        {
                            LOG_INFO("[MoE] Post-warmup setup complete"
                                     << (controller->hasReplicas()
                                         ? " (with per-token replica dispatch)"
                                         : " (local repack only)"));
                        }
                    });
                    // No per-step rebalancing — the post-warmup placement
                    // is used for all benchmark iterations.
                }
            }
        }

        BenchmarkResult result = benchmark.run(ctx.config);
        benchmark.printResults(result);
        MoEExpertOverlayProfiler::flush();

        // Log MoE histogram if controller is active
        if (auto* orch_runner = dynamic_cast<OrchestrationRunner*>(runner.get()))
        {
            if (auto* controller = orch_runner->moeRebalanceController())
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
