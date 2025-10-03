#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include "src/kernels/common/softmax_core.h"
#include "src/utils/debug_env.h"

using namespace llaminar::kernels;

struct BenchResult
{
    double ms;
    double gbytes;
    double calls;
};

static BenchResult bench_case(int rows, int cols, int iters, bool causal, float scale)
{
    std::vector<float> data((size_t)rows * cols);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-4.f, 4.f);
    for (auto &v : data)
        v = dist(rng);
    SoftmaxRowArgs args;
    args.scores = data.data();
    args.rows = rows;
    args.cols = cols;
    args.causal = causal;
    args.scale = scale;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        softmax_row_major(args);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    // Rough IO: read + write array each pass (3 passes effectively) ~ 3 * size * sizeof(float)
    double bytes = (double)rows * cols * sizeof(float) * 3;
    double gbytes = (bytes * iters) / 1e9;
    return {ms, gbytes, (double)iters};
}

int main(int argc, char **argv)
{
    const auto &env = llaminar::debugEnv();
    struct Shape
    {
        int rows;
        int cols;
        int iters;
        bool causal;
        float scale;
    };
    std::vector<Shape> shapes = {
        {1, 4096, 100000, false, 1.0f},
        {1, 8192, 50000, false, 1.0f},
        {4, 4096, 40000, false, 1.0f},
        {16, 4096, 10000, false, 1.0f},
        {32, 4096, 5000, false, 1.0f},
        {32, 8192, 3000, false, 1.0f},
        {64, 8192, 1500, false, 1.0f},
        {64, 8192, 1500, true, 1.0f},
        {64, 8192, 1500, true, 0.8f}};
    std::cout << "# Softmax Bench (force_scalar=" << env.softmax.force_scalar
              << " parallel_elems_threshold=" << env.softmax.parallel_elems_threshold
              << ")" << std::endl;
    std::cout << "rows,cols,iters,causal,scale,ms,total_GB,GB/s" << std::endl;
    for (auto sh : shapes)
    {
        auto r = bench_case(sh.rows, sh.cols, sh.iters, sh.causal, sh.scale);
        double gbps = r.gbytes / (r.ms / 1000.0);
        std::cout << sh.rows << ',' << sh.cols << ',' << sh.iters << ',' << (sh.causal ? 1 : 0) << ',' << sh.scale << ','
                  << r.ms << ',' << r.gbytes << ',' << gbps << std::endl;
    }
    return 0;
}
