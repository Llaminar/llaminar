/**
 * @file BenchmarkMode.cpp
 * @brief Benchmark mode (--benchmark)
 */

#include "app/modes/BenchmarkMode.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/moe/MoERebalanceController.h"
#include "utils/Logger.h"
#include "utils/BenchmarkRunner.h"
#include <mpi.h>

namespace llaminar2
{

    bool BenchmarkMode::matches(const OrchestrationConfig &config) const
    {
        return config.benchmark_mode;
    }

    int BenchmarkMode::execute(AppContext &ctx)
    {
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running benchmark mode...");
        }

        auto adapter = std::make_shared<InferenceRunnerAdapter>(runner.get());

        BenchmarkRunner benchmark(adapter, tokenizer, mpi_ctx);

        // Set up MoE expert rebalancing callback (invoked after warmup)
        if (auto* orch_runner = dynamic_cast<OrchestrationRunner*>(runner.get()))
        {
            if (auto* controller = orch_runner->moeRebalanceController())
            {
                if (controller->mode() == MoERebalanceMode::DYNAMIC)
                {
                    benchmark.setPostWarmupCallback([orch_runner, controller, &mpi_ctx]() {
                        controller->logHistogramSummary();
                        controller->rebalanceLPT();
                        int socket_id = mpi_ctx->rank(); // rank == socket for dual-socket
                        auto masks = controller->computeExpertMasks(socket_id);
                        orch_runner->applyMoEExpertMasks(masks);
                        if (mpi_ctx->rank() == 0)
                        {
                            LOG_INFO("[BenchmarkMode] Expert rebalancing applied after warmup");
                        }
                    });
                }
            }
        }

        BenchmarkResult result = benchmark.run(ctx.config);
        benchmark.printResults(result);

        // Log MoE histogram if controller is active
        if (auto* orch_runner = dynamic_cast<OrchestrationRunner*>(runner.get()))
        {
            if (auto* controller = orch_runner->moeRebalanceController())
            {
                controller->logHistogramSummary();
            }
        }

        runner->shutdown();
        MPI_Finalize();
        return result.success ? 0 : 1;
    }

} // namespace llaminar2
