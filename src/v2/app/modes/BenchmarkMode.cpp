/**
 * @file BenchmarkMode.cpp
 * @brief Benchmark mode (--benchmark)
 */

#include "app/modes/BenchmarkMode.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
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
        BenchmarkResult result = benchmark.run(ctx.config);
        benchmark.printResults(result);

        runner->shutdown();
        MPI_Finalize();
        return result.success ? 0 : 1;
    }

} // namespace llaminar2
