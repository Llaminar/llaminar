#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <mpi.h>
#include "src/kernels/common/softmax_core.h"
#include "src/utils/debug_env.h"

using namespace llaminar::kernels;

struct DistBenchCase
{
    int rows;
    int cols;
    bool causal;
    float scale;
    int iters;
};

static void run_case(const DistBenchCase &bc, int rank, int world)
{
    // Column partition: contiguous blocks; distribute remainder to lower ranks
    int base = bc.cols / world;
    int rem = bc.cols % world;
    int local_cols = base + (rank < rem ? 1 : 0);
    int offset = 0;
    for (int r = 0; r < rank; ++r)
        offset += base + (r < rem ? 1 : 0);

    std::vector<float> local((size_t)bc.rows * local_cols);
    // Initialize replicated full logits on root then scatter manually
    std::vector<float> full;
    if (rank == 0)
    {
        full.resize((size_t)bc.rows * bc.cols);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-4.f, 4.f);
        for (auto &v : full)
            v = dist(rng);
    }
    // Broadcast full then slice locally (simpler than building custom scatter)
    if (rank == 0)
    {
        for (int r = 1; r < world; ++r)
            MPI_Send(full.data(), (int)full.size(), MPI_FLOAT, r, 0, MPI_COMM_WORLD);
    }
    else
    {
        full.resize((size_t)bc.rows * bc.cols);
        MPI_Status st{};
        MPI_Recv(full.data(), (int)full.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &st);
    }
    for (int r = 0; r < bc.rows; ++r)
    {
        std::memcpy(local.data() + (size_t)r * local_cols,
                    full.data() + (size_t)r * bc.cols + offset,
                    sizeof(float) * local_cols);
    }

    DistributedSoftmaxCtx ctx;
    ctx.comm = MPI_COMM_WORLD;
    ctx.world_size = world;
    ctx.rank = rank;
    ctx.use_barrier = false;
    SoftmaxRowArgs args;
    args.scores = local.data();
    args.rows = bc.rows;
    args.cols = local_cols;
    args.causal = bc.causal;
    args.scale = bc.scale;

    // Warmup
    softmax_distributed(args, bc.rows, 0, bc.cols, offset, ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bc.iters; ++i)
    {
        softmax_distributed(args, bc.rows, 0, bc.cols, offset, ctx);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    // Rough IO: two passes read + write local slice; approximate 3 * local slice bytes per iter
    double bytes = 3.0 * (double)bc.rows * local_cols * sizeof(float) * bc.iters;
    double gbytes = bytes / 1e9;
    // Gather local times? We use barrier around timing so ms identical across ranks (rank 0 authoritative)
    // Compute global bytes (sum of local) deterministically: each rank can compute its own local_cols; simplest is reduce.
    double local_bytes = 3.0 * (double)bc.rows * local_cols * sizeof(float) * bc.iters;
    double global_bytes = 0.0;
    MPI_Reduce(&local_bytes, &global_bytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        const auto &env = llaminar::debugEnv().softmax;
        std::string mode;
        if (env.fast_exp && env.dist_recompute)
            mode = "fast_exp+recompute";
        else if (env.fast_exp)
            mode = "fast_exp";
        else if (env.dist_recompute || (env.dist_recompute_threshold > 0 && (size_t)bc.rows * bc.cols >= (size_t)env.dist_recompute_threshold))
            mode = "recompute";
        else
            mode = "baseline";
        double global_gbytes = global_bytes / 1e9;
        double gbps_local = gbytes / (ms / 1000.0);
        double gbps_global = global_gbytes / (ms / 1000.0);
        std::cout << bc.rows << ',' << bc.cols << ',' << local_cols << ',' << bc.causal << ',' << bc.scale << ','
                  << bc.iters << ',' << ms << ',' << gbytes << ',' << gbps_local << ','
                  << global_gbytes << ',' << gbps_global << ',' << mode << '\n';
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (rank == 0)
    {
        std::cout << "# Distributed Softmax Bench (world=" << world << ")\n";
        std::cout << "rows,global_cols,local_cols,causal,scale,iters,ms,local_GB,GB/s_local,global_GB,GB/s_global,mode" << std::endl;
    }
    std::vector<DistBenchCase> cases = {
        {16, 4096, false, 1.0f, 4000},
        {32, 4096, false, 1.0f, 2000},
        {32, 8192, false, 1.0f, 1500},
        {32, 8192, true, 1.0f, 1500},
        {64, 8192, true, 0.8f, 1000}};
    for (auto &bc : cases)
        run_case(bc, rank, world);
    MPI_Finalize();
    return 0;
}
