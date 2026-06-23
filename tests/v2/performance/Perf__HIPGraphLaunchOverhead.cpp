/**
 * @file Perf__HIPGraphLaunchOverhead.cpp
 * @brief Performance benchmark: HIP graph capture + launch overhead vs direct dispatch
 *
 * **Motivation**: Enabling LLAMINAR_GPU_GRAPHS=1 on MI50 causes a ~12% decode
 * throughput regression (71.4 → 63.5 tok/s). This test isolates whether the
 * overhead comes from:
 *   A) hipGraphLaunch() itself (driver overhead to walk 700+ graph nodes)
 *   B) The stream switch (capture_stream vs default_stream)
 *   C) CPU-side coherence / bookkeeping per-segment
 *   D) Something about the captured kernels executing differently
 *
 * **Test Plan**:
 *   1. Capture graphs of varying node counts (1, 10, 50, 100, 338, 700+)
 *      using trivial HIP kernels (vector add — ~5 μs each)
 *   2. Measure:
 *      - Direct kernel dispatch (no graph): N kernels on default stream
 *      - Graph launch + hipStreamSynchronize: same N kernels via graph replay
 *      - Empty graph launch (just launch overhead, no real work)
 *   3. Report per-call and total overhead in microseconds
 *
 * **Expected**: If hipGraphLaunch has ~2ms overhead for 732 nodes, that
 * explains the regression: 1/71.4s = 14ms/tok, +2ms = 16ms → 62.5 tok/s.
 *
 * @author GitHub Copilot
 * @date February 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "fort.hpp"

// ============================================================================
// HIP Graph Overhead Benchmark
// ============================================================================

#ifdef HAVE_ROCM

// Use EXPECT (not ASSERT) so we can use HIP_CHECK in non-void helper functions.
// Fatal failures are still caught by GTest at the call-site level.
#define HIP_CHECK(call)                                                         \
    do                                                                          \
    {                                                                           \
        hipError_t err = (call);                                                \
        EXPECT_EQ(err, hipSuccess) << #call << " failed: " << hipGetErrorString(err); \
    } while (0)

// Trivial kernel — just enough work to be a real kernel launch but minimal compute
__global__ void vector_add_kernel(float *C, const float *A, const float *B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// Empty kernel — absolute minimum kernel node
__global__ void nop_kernel()
{
    // intentionally empty
}

namespace
{

// ============================================================================
// Timing Utilities
// ============================================================================

struct BenchResult
{
    double mean_us = 0.0;
    double median_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double stddev_us = 0.0;
    int iterations = 0;
    int kernel_count = 0;
};

BenchResult computeStats(std::vector<double> &samples, int kernel_count)
{
    BenchResult r;
    r.iterations = static_cast<int>(samples.size());
    r.kernel_count = kernel_count;
    if (samples.empty())
        return r;

    std::sort(samples.begin(), samples.end());
    r.min_us = samples.front();
    r.max_us = samples.back();
    r.median_us = samples[samples.size() / 2];
    r.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

    double var = 0.0;
    for (double s : samples)
    {
        double d = s - r.mean_us;
        var += d * d;
    }
    r.stddev_us = std::sqrt(var / samples.size());
    return r;
}

// ============================================================================
// Benchmark Parameters
// ============================================================================

constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS = 100;
constexpr int VECTOR_N = 1024;  // Small vector — kernel ~5μs, dominated by launch
constexpr int BLOCK_SIZE = 256;

// Node counts to test: covers small graphs up to Qwen2.5-7B decode graph (732 nodes)
const std::vector<int> NODE_COUNTS = {1, 5, 10, 28, 50, 100, 200, 338, 500, 732};

// ============================================================================
// Test Fixture
// ============================================================================

class Perf__HIPGraphLaunchOverhead : public ::testing::Test
{
protected:
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    hipStream_t stream_ = nullptr;

    void SetUp() override
    {
        // Allocate small GPU buffers for vector add
        HIP_CHECK(hipMalloc(&d_A, VECTOR_N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B, VECTOR_N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_C, VECTOR_N * sizeof(float)));

        // Initialize with some data
        std::vector<float> host(VECTOR_N, 1.0f);
        HIP_CHECK(hipMemcpy(d_A, host.data(), VECTOR_N * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, host.data(), VECTOR_N * sizeof(float), hipMemcpyHostToDevice));

        // Create a non-blocking stream (same as capture path uses)
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));

        // Warm up the GPU
        int grid = (VECTOR_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < 5; i++)
        {
            vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
        }
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    void TearDown() override
    {
        if (stream_)
        {
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
        }
        (void)hipFree(d_A);
        (void)hipFree(d_B);
        (void)hipFree(d_C);
    }

    // -----------------------------------------------------------------------
    // Benchmark: Direct kernel dispatch (no graph)
    // -----------------------------------------------------------------------
    BenchResult benchDirectDispatch(int num_kernels)
    {
        int grid = (VECTOR_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++)
        {
            for (int k = 0; k < num_kernels; k++)
            {
                vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
            }
            (void)hipStreamSynchronize(stream_);
        }

        // Benchmark
        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < num_kernels; k++)
            {
                vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
            }
            (void)hipStreamSynchronize(stream_);
            auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
        return computeStats(samples, num_kernels);
    }

    // -----------------------------------------------------------------------
    // Benchmark: Graph capture + launch (real kernels)
    // -----------------------------------------------------------------------
    BenchResult benchGraphLaunch(int num_kernels)
    {
        int grid = (VECTOR_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Capture the graph
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;

        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed));
        for (int k = 0; k < num_kernels; k++)
        {
            vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
        }
        HIP_CHECK(hipStreamEndCapture(stream_, &graph));
        EXPECT_NE(graph, nullptr);

        // Query node count
        size_t node_count = 0;
        (void)hipGraphGetNodes(graph, nullptr, &node_count);

        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        // Warmup launches
        for (int w = 0; w < WARMUP_ITERS; w++)
        {
            (void)hipGraphLaunch(exec, stream_);
            (void)hipStreamSynchronize(stream_);
        }

        // Benchmark
        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            (void)hipGraphLaunch(exec, stream_);
            (void)hipStreamSynchronize(stream_);
            auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        auto result = computeStats(samples, num_kernels);

        // Cleanup
        (void)hipGraphExecDestroy(exec);
        (void)hipGraphDestroy(graph);

        return result;
    }

    // -----------------------------------------------------------------------
    // Benchmark: Graph launch with nop kernels (pure launch overhead)
    // -----------------------------------------------------------------------
    BenchResult benchGraphLaunchNop(int num_kernels)
    {
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;

        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed));
        for (int k = 0; k < num_kernels; k++)
        {
            nop_kernel<<<1, 1, 0, stream_>>>();
        }
        HIP_CHECK(hipStreamEndCapture(stream_, &graph));
        EXPECT_NE(graph, nullptr);

        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++)
        {
            (void)hipGraphLaunch(exec, stream_);
            (void)hipStreamSynchronize(stream_);
        }

        // Benchmark
        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            (void)hipGraphLaunch(exec, stream_);
            (void)hipStreamSynchronize(stream_);
            auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        auto result = computeStats(samples, num_kernels);

        (void)hipGraphExecDestroy(exec);
        (void)hipGraphDestroy(graph);
        return result;
    }

    // -----------------------------------------------------------------------
    // Benchmark: Direct dispatch of nop kernels (baseline for nop path)
    // -----------------------------------------------------------------------
    BenchResult benchDirectNop(int num_kernels)
    {
        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++)
        {
            for (int k = 0; k < num_kernels; k++)
            {
                nop_kernel<<<1, 1, 0, stream_>>>();
            }
            (void)hipStreamSynchronize(stream_);
        }

        // Benchmark
        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < num_kernels; k++)
            {
                nop_kernel<<<1, 1, 0, stream_>>>();
            }
            (void)hipStreamSynchronize(stream_);
            auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
        return computeStats(samples, num_kernels);
    }

    // -----------------------------------------------------------------------
    // Benchmark: hipDeviceSynchronize() vs hipStreamSynchronize() overhead
    // -----------------------------------------------------------------------
    BenchResult benchDeviceSync(int num_kernels)
    {
        int grid = (VECTOR_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++)
        {
            for (int k = 0; k < num_kernels; k++)
            {
                vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
            }
            (void)hipDeviceSynchronize();
        }

        // Benchmark
        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < num_kernels; k++)
            {
                vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
            }
            (void)hipDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
        return computeStats(samples, num_kernels);
    }

    // -----------------------------------------------------------------------
    // Benchmark: Graph launch with hipDeviceSynchronize (matches replay code)
    // -----------------------------------------------------------------------
    BenchResult benchGraphLaunchDeviceSync(int num_kernels)
    {
        int grid = (VECTOR_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;

        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed));
        for (int k = 0; k < num_kernels; k++)
        {
            vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
        }
        HIP_CHECK(hipStreamEndCapture(stream_, &graph));
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        // Warmup
        for (int w = 0; w < WARMUP_ITERS; w++)
        {
            (void)hipGraphLaunch(exec, stream_);
            (void)hipDeviceSynchronize();
        }

        // Benchmark
        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            (void)hipGraphLaunch(exec, stream_);
            (void)hipDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        auto result = computeStats(samples, num_kernels);
        (void)hipGraphExecDestroy(exec);
        (void)hipGraphDestroy(graph);
        return result;
    }

    // -----------------------------------------------------------------------
    // Benchmark: Graph capture cost (one-time)
    // -----------------------------------------------------------------------
    BenchResult benchGraphCaptureCost(int num_kernels)
    {
        int grid = (VECTOR_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Warmup
        for (int w = 0; w < 3; w++)
        {
            hipGraph_t g = nullptr;
            hipGraphExec_t e = nullptr;
            (void)hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed);
            for (int k = 0; k < num_kernels; k++)
            {
                vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
            }
            (void)hipStreamEndCapture(stream_, &g);
            (void)hipGraphInstantiate(&e, g, nullptr, nullptr, 0);
            (void)hipGraphExecDestroy(e);
            (void)hipGraphDestroy(g);
        }

        // Benchmark: full capture + instantiate cycle
        std::vector<double> samples;
        samples.reserve(20); // Fewer iters since capture is expensive
        for (int i = 0; i < 20; i++)
        {
            hipGraph_t g = nullptr;
            hipGraphExec_t e = nullptr;

            auto start = std::chrono::high_resolution_clock::now();
            (void)hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed);
            for (int k = 0; k < num_kernels; k++)
            {
                vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream_>>>(d_C, d_A, d_B, VECTOR_N);
            }
            (void)hipStreamEndCapture(stream_, &g);
            (void)hipGraphInstantiate(&e, g, nullptr, nullptr, 0);
            auto end = std::chrono::high_resolution_clock::now();

            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());

            (void)hipGraphExecDestroy(e);
            (void)hipGraphDestroy(g);
        }
        return computeStats(samples, num_kernels);
    }
};

// ============================================================================
// TEST 1: Launch overhead scaling by graph size
// ============================================================================

TEST_F(Perf__HIPGraphLaunchOverhead, LaunchOverheadByNodeCount)
{
    printf("\n");

    // ---- Main comparison table ----
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    table << fort::header
          << "Kernels"
          << "Direct\n(median μs)"
          << "Graph\n(median μs)"
          << "Overhead\n(μs)"
          << "Overhead\n(%)"
          << "Per-kernel\nDirect (μs)"
          << "Per-kernel\nGraph (μs)"
          << fort::endr;

    for (int i = 1; i <= 6; i++)
        table.column(i).set_cell_text_align(fort::text_align::right);

    for (int n : NODE_COUNTS)
    {
        auto direct = benchDirectDispatch(n);
        auto graph = benchGraphLaunch(n);
        double overhead_us = graph.median_us - direct.median_us;
        double overhead_pct = (direct.median_us > 0)
                                  ? 100.0 * overhead_us / direct.median_us
                                  : 0.0;

        char buf_direct[32], buf_graph[32], buf_oh[32], buf_pct[32];
        char buf_pk_direct[32], buf_pk_graph[32];
        snprintf(buf_direct, sizeof(buf_direct), "%.1f", direct.median_us);
        snprintf(buf_graph, sizeof(buf_graph), "%.1f", graph.median_us);
        snprintf(buf_oh, sizeof(buf_oh), "%+.1f", overhead_us);
        snprintf(buf_pct, sizeof(buf_pct), "%+.1f%%", overhead_pct);
        snprintf(buf_pk_direct, sizeof(buf_pk_direct), "%.2f", direct.median_us / n);
        snprintf(buf_pk_graph, sizeof(buf_pk_graph), "%.2f", graph.median_us / n);

        table << std::to_string(n)
              << buf_direct << buf_graph << buf_oh << buf_pct
              << buf_pk_direct << buf_pk_graph
              << fort::endr;
    }

    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        HIP GRAPH LAUNCH OVERHEAD: Direct vs Graph (vector_add)         ║\n");
    printf("║        Sync: hipStreamSynchronize | %d warmup, %d bench iters          ║\n",
           WARMUP_ITERS, BENCH_ITERS);
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("%s\n", table.to_string().c_str());
}

// ============================================================================
// TEST 2: Nop kernel graph — pure launch + sync overhead
// ============================================================================

TEST_F(Perf__HIPGraphLaunchOverhead, NopKernelGraphOverhead)
{
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    table << fort::header
          << "Nodes"
          << "Direct Nop\n(median μs)"
          << "Graph Nop\n(median μs)"
          << "Overhead\n(μs)"
          << "Overhead\n(%)"
          << fort::endr;

    for (int i = 1; i <= 4; i++)
        table.column(i).set_cell_text_align(fort::text_align::right);

    for (int n : NODE_COUNTS)
    {
        auto direct = benchDirectNop(n);
        auto graph = benchGraphLaunchNop(n);
        double overhead_us = graph.median_us - direct.median_us;
        double overhead_pct = (direct.median_us > 0)
                                  ? 100.0 * overhead_us / direct.median_us
                                  : 0.0;

        char buf[5][32];
        snprintf(buf[0], 32, "%.1f", direct.median_us);
        snprintf(buf[1], 32, "%.1f", graph.median_us);
        snprintf(buf[2], 32, "%+.1f", overhead_us);
        snprintf(buf[3], 32, "%+.1f%%", overhead_pct);

        table << std::to_string(n) << buf[0] << buf[1] << buf[2] << buf[3] << fort::endr;
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║            HIP GRAPH LAUNCH OVERHEAD: Nop Kernels (pure overhead)      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("%s\n", table.to_string().c_str());
}

// ============================================================================
// TEST 3: Sync method comparison — hipStreamSynchronize vs hipDeviceSynchronize
// ============================================================================

TEST_F(Perf__HIPGraphLaunchOverhead, SyncMethodComparison)
{
    // The replay path currently uses hipDeviceSynchronize (gpu_ctx->synchronize()).
    // Compare with hipStreamSynchronize to see if that's contributing overhead.
    const std::vector<int> counts = {1, 28, 100, 338, 732};

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    table << fort::header
          << "Kernels"
          << "Direct\nStreamSync"
          << "Direct\nDeviceSync"
          << "Graph\nStreamSync"
          << "Graph\nDeviceSync"
          << "Δ Sync\nDirect (μs)"
          << "Δ Sync\nGraph (μs)"
          << fort::endr;

    for (int i = 1; i <= 6; i++)
        table.column(i).set_cell_text_align(fort::text_align::right);

    for (int n : counts)
    {
        auto direct_stream = benchDirectDispatch(n);
        auto direct_device = benchDeviceSync(n);
        auto graph_stream = benchGraphLaunch(n);
        auto graph_device = benchGraphLaunchDeviceSync(n);

        char buf[7][32];
        snprintf(buf[0], 32, "%.1f", direct_stream.median_us);
        snprintf(buf[1], 32, "%.1f", direct_device.median_us);
        snprintf(buf[2], 32, "%.1f", graph_stream.median_us);
        snprintf(buf[3], 32, "%.1f", graph_device.median_us);
        snprintf(buf[4], 32, "%+.1f", direct_device.median_us - direct_stream.median_us);
        snprintf(buf[5], 32, "%+.1f", graph_device.median_us - graph_stream.median_us);

        table << std::to_string(n) << buf[0] << buf[1] << buf[2] << buf[3] << buf[4] << buf[5]
              << fort::endr;
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        SYNC METHOD COMPARISON: StreamSync vs DeviceSync                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("%s\n", table.to_string().c_str());
}

// ============================================================================
// TEST 4: Graph capture + instantiate cost (one-time)
// ============================================================================

TEST_F(Perf__HIPGraphLaunchOverhead, CaptureCost)
{
    const std::vector<int> counts = {1, 10, 50, 100, 338, 732};

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    table << fort::header
          << "Kernels"
          << "Capture+Inst\n(median μs)"
          << "Capture+Inst\n(min μs)"
          << "Capture+Inst\n(max μs)"
          << fort::endr;

    for (int i = 1; i <= 3; i++)
        table.column(i).set_cell_text_align(fort::text_align::right);

    for (int n : counts)
    {
        auto r = benchGraphCaptureCost(n);
        char buf[4][32];
        snprintf(buf[0], 32, "%.1f", r.median_us);
        snprintf(buf[1], 32, "%.1f", r.min_us);
        snprintf(buf[2], 32, "%.1f", r.max_us);

        table << std::to_string(n) << buf[0] << buf[1] << buf[2] << fort::endr;
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        GRAPH CAPTURE + INSTANTIATE COST (one-time per config)          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("%s\n", table.to_string().c_str());
}

// ============================================================================
// TEST 5: Decode budget analysis — how much overhead is acceptable?
// ============================================================================

TEST_F(Perf__HIPGraphLaunchOverhead, DecodeBudgetAnalysis)
{
    // For decode at ~71 tok/s, each token takes 14.08ms budget.
    // Measure the overhead at the production graph size (338 stages ≈ 732 nodes)
    // and project how much throughput we'd lose.

    constexpr int PRODUCTION_KERNELS = 338;

    auto direct = benchDirectDispatch(PRODUCTION_KERNELS);
    auto graph = benchGraphLaunch(PRODUCTION_KERNELS);
    auto graph_dev_sync = benchGraphLaunchDeviceSync(PRODUCTION_KERNELS);
    auto nop_graph = benchGraphLaunchNop(PRODUCTION_KERNELS);

    double target_tok_s = 71.4;
    double budget_us = 1e6 / target_tok_s; // μs per token

    double graph_overhead_us = graph.median_us - direct.median_us;
    double projected_tok_s = 1e6 / (budget_us + graph_overhead_us);
    double throughput_loss_pct = 100.0 * (target_tok_s - projected_tok_s) / target_tok_s;

    double dev_sync_overhead = graph_dev_sync.median_us - graph.median_us;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║            DECODE BUDGET ANALYSIS (Qwen2.5-7B Q8_0 @ MI50)             ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                        ║\n");
    printf("║  Production config: %d stages → ~732 HIP graph nodes               ║\n", PRODUCTION_KERNELS);
    printf("║                                                                        ║\n");
    printf("║  Direct dispatch (stream sync):     %8.1f μs (median)               ║\n", direct.median_us);
    printf("║  Graph launch (stream sync):        %8.1f μs (median)               ║\n", graph.median_us);
    printf("║  Graph launch (device sync):        %8.1f μs (median)               ║\n", graph_dev_sync.median_us);
    printf("║  Nop graph launch (pure overhead):  %8.1f μs (median)               ║\n", nop_graph.median_us);
    printf("║                                                                        ║\n");
    printf("║  ── Overhead breakdown ──                                              ║\n");
    printf("║  Graph vs Direct (launch overhead):  %+7.1f μs                        ║\n", graph_overhead_us);
    printf("║  DeviceSync vs StreamSync (extra):   %+7.1f μs                        ║\n", dev_sync_overhead);
    printf("║  Pure driver walk (nop graph):       %7.1f μs                         ║\n", nop_graph.median_us);
    printf("║                                                                        ║\n");
    printf("║  ── Throughput projection ──                                           ║\n");
    printf("║  Decode budget at %.1f tok/s:       %8.1f μs/token                  ║\n", target_tok_s, budget_us);
    printf("║  Projected with graph overhead:      %8.1f μs/token                  ║\n", budget_us + graph_overhead_us);
    printf("║  Projected throughput:               %8.1f tok/s                      ║\n", projected_tok_s);
    printf("║  Throughput loss:                    %+7.1f%%                           ║\n", -throughput_loss_pct);
    printf("║                                                                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");
}

} // anonymous namespace

#else // !HAVE_ROCM

TEST(Perf__HIPGraphLaunchOverhead, SkipNoROCm)
{
    GTEST_SKIP() << "ROCm not available";
}

#endif // HAVE_ROCM
