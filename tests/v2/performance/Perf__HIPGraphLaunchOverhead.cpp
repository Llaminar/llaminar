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
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <thread>
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
const std::vector<int> NODE_COUNTS = {1, 5, 10, 19, 28, 50, 100, 200, 338, 500, 732};

/**
 * @brief One setup-owned retained graph used by the multi-device launch test.
 *
 * The production heterogeneous MoE endpoint retains one graph per participant
 * device.  Keeping every stream, graph executable, and completion event alive
 * here makes the timed loop measure only replay submission and publication.
 */
struct RetainedDeviceGraph
{
    int device = -1;                 ///< HIP ordinal owned by this endpoint.
    hipStream_t stream = nullptr;    ///< Exact non-default replay stream.
    hipGraph_t graph = nullptr;      ///< Captured 19-node graph definition.
    hipGraphExec_t executable = nullptr; ///< Instantiated replay authority.
    hipEvent_t completion = nullptr; ///< Exact event published after replay.
    hipError_t worker_error = hipSuccess; ///< First asynchronous worker error.
};

/**
 * @brief One captured graph definition used to measure executable publication.
 *
 * Production heterogeneous ExpertOverlay retains many independently launchable
 * graph segments per physical bucket.  The captured definition and explicit
 * stream remain stable while the benchmark creates a family of executable
 * instances, reproducing the driver-side registry pressure without loading a
 * model or conflating graph launch latency with graph publication latency.
 */
struct CapturedInstantiationTemplate
{
    int device = -1;             ///< HIP ordinal that owns every native handle.
    hipStream_t stream = nullptr; ///< Exact non-default capture stream.
    hipGraph_t graph = nullptr;  ///< Reusable captured graph definition.
};

/**
 * @brief Destroy one graph template on its immutable owner device.
 * @param graph Template whose native resources are released.
 */
void destroyCapturedInstantiationTemplate(
    CapturedInstantiationTemplate &graph) noexcept
{
    if (graph.device >= 0)
        (void)hipSetDevice(graph.device);
    if (graph.graph)
        (void)hipGraphDestroy(graph.graph);
    if (graph.stream)
        (void)hipStreamDestroy(graph.stream);
    graph = {};
}

/**
 * @brief Record a fixed-width kernel graph without instantiating it.
 * @param graph Destination template populated on success.
 * @param device HIP ordinal that owns the graph.
 * @param node_count Number of ordered kernel nodes to record.
 * @return First failing HIP status, or `hipSuccess`.
 */
hipError_t captureInstantiationTemplate(
    CapturedInstantiationTemplate &graph,
    int device,
    int node_count)
{
    graph.device = device;
    hipError_t status = hipSetDevice(device);
    if (status != hipSuccess)
        return status;
    status = hipStreamCreateWithFlags(&graph.stream, hipStreamNonBlocking);
    if (status != hipSuccess)
        return status;
    status = hipStreamBeginCapture(
        graph.stream,
        hipStreamCaptureModeRelaxed);
    if (status != hipSuccess)
        return status;
    for (int node = 0; node < node_count; ++node)
        nop_kernel<<<1, 1, 0, graph.stream>>>();
    return hipStreamEndCapture(graph.stream, &graph.graph);
}

/**
 * @brief Result of one process-wide graph-executable publication wave.
 */
struct InstantiationWaveResult
{
    hipError_t status = hipSuccess; ///< First driver error across all devices.
    double elapsed_ms = 0.0;        ///< Complete retained-family wall time.
};

/**
 * @brief Instantiate and retain the same executable cardinality per device.
 *
 * The concurrent form starts one worker per device at the same barrier.  The
 * serial form performs identical driver calls and retains the same handles,
 * but publishes them in deterministic device-major order.  All handles remain
 * alive until the timed wave has finished so the measurement includes the
 * native registry pressure present in a serving graph family.
 *
 * @param templates One captured definition per participating device.
 * @param executable_count Number of retained executables per device.
 * @param concurrent Whether devices publish in parallel or serial order.
 * @return Driver status and elapsed wall time; all executables are destroyed
 *         before the function returns.
 */
InstantiationWaveResult instantiateRetainedFamily(
    std::span<const CapturedInstantiationTemplate> templates,
    int executable_count,
    bool concurrent)
{
    InstantiationWaveResult result;
    std::vector<std::vector<hipGraphExec_t>> executables(
        templates.size(),
        std::vector<hipGraphExec_t>(
            static_cast<std::size_t>(executable_count), nullptr));
    std::vector<hipError_t> worker_status(
        templates.size(), hipSuccess);

    const auto instantiate_device = [&](std::size_t index)
    {
        const auto &graph = templates[index];
        hipError_t status = hipSetDevice(graph.device);
        for (int executable = 0;
             status == hipSuccess && executable < executable_count;
             ++executable)
        {
            status = hipGraphInstantiate(
                &executables[index][static_cast<std::size_t>(executable)],
                graph.graph,
                nullptr,
                nullptr,
                0);
        }
        worker_status[index] = status;
    };

    const auto begin = std::chrono::steady_clock::now();
    if (concurrent)
    {
        std::atomic<std::size_t> ready{0u};
        std::atomic<bool> start{false};
        std::vector<std::thread> workers;
        workers.reserve(templates.size());
        for (std::size_t index = 0u; index < templates.size(); ++index)
        {
            workers.emplace_back(
                [&, index]()
                {
                    ready.fetch_add(1u, std::memory_order_release);
                    while (!start.load(std::memory_order_acquire))
                        std::this_thread::yield();
                    instantiate_device(index);
                });
        }
        while (ready.load(std::memory_order_acquire) != templates.size())
            std::this_thread::yield();
        start.store(true, std::memory_order_release);
        for (auto &worker : workers)
            worker.join();
    }
    else
    {
        for (std::size_t index = 0u; index < templates.size(); ++index)
            instantiate_device(index);
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin)
                            .count();

    for (std::size_t index = 0u; index < templates.size(); ++index)
    {
        if (result.status == hipSuccess &&
            worker_status[index] != hipSuccess)
        {
            result.status = worker_status[index];
        }
        (void)hipSetDevice(templates[index].device);
        for (hipGraphExec_t executable : executables[index])
        {
            if (executable)
                (void)hipGraphExecDestroy(executable);
        }
    }
    return result;
}

/**
 * @brief Destroy a retained endpoint on the device that owns its resources.
 * @param endpoint Endpoint whose setup-owned resources are released.
 */
void destroyRetainedDeviceGraph(RetainedDeviceGraph &endpoint) noexcept
{
    if (endpoint.device >= 0)
        (void)hipSetDevice(endpoint.device);
    if (endpoint.completion)
        (void)hipEventDestroy(endpoint.completion);
    if (endpoint.executable)
        (void)hipGraphExecDestroy(endpoint.executable);
    if (endpoint.graph)
        (void)hipGraphDestroy(endpoint.graph);
    if (endpoint.stream)
        (void)hipStreamDestroy(endpoint.stream);
    endpoint = {};
}

/**
 * @brief Capture and instantiate one fixed-size no-op endpoint graph.
 * @param endpoint Destination owner populated on success.
 * @param device HIP device ordinal assigned to the endpoint.
 * @param node_count Number of kernel nodes captured in the retained graph.
 * @return First HIP status that fails, or `hipSuccess`.
 */
hipError_t initializeRetainedDeviceGraph(
    RetainedDeviceGraph &endpoint,
    int device,
    int node_count)
{
    endpoint.device = device;
    hipError_t status = hipSetDevice(device);
    if (status != hipSuccess)
        return status;
    status = hipStreamCreateWithFlags(&endpoint.stream, hipStreamNonBlocking);
    if (status != hipSuccess)
        return status;
    status = hipEventCreateWithFlags(
        &endpoint.completion,
        hipEventDisableTiming);
    if (status != hipSuccess)
        return status;

    status = hipStreamBeginCapture(
        endpoint.stream,
        hipStreamCaptureModeRelaxed);
    if (status != hipSuccess)
        return status;
    for (int node = 0; node < node_count; ++node)
        nop_kernel<<<1, 1, 0, endpoint.stream>>>();
    status = hipStreamEndCapture(endpoint.stream, &endpoint.graph);
    if (status != hipSuccess)
        return status;
    return hipGraphInstantiate(
        &endpoint.executable,
        endpoint.graph,
        nullptr,
        nullptr,
        0);
}

/**
 * @brief Allocation-free persistent fan-out for retained HIP graph submission.
 *
 * Each worker binds one HIP device exactly once, then waits for a monotonically
 * increasing generation.  A generation only launches its endpoint graph and
 * records the exact completion event; the coordinator performs the later host
 * fence after every sibling has been submitted.  This is the same two-wave
 * ordering required by the ExpertOverlay participant graph, without allocating
 * futures, callables, or threads inside the measured path.
 */
class PersistentHIPGraphLaunchWave final
{
public:
    /** @brief Start one permanent worker for every retained endpoint. */
    explicit PersistentHIPGraphLaunchWave(
        std::vector<RetainedDeviceGraph> &endpoints)
        : endpoints_(endpoints)
    {
        workers_.reserve(endpoints_.size());
        for (size_t index = 0; index < endpoints_.size(); ++index)
        {
            workers_.emplace_back(
                [this, index]()
                {
                    workerLoop(index);
                });
        }

        // Construction does not return until every worker has installed its
        // immutable device context. No timed generation pays this setup cost.
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cv_.wait(
            lock,
            [this]()
            {
                return ready_count_ == endpoints_.size();
            });
    }

    /** @brief Stop and join every permanent worker. */
    ~PersistentHIPGraphLaunchWave()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        dispatch_cv_.notify_all();
        for (auto &worker : workers_)
        {
            if (worker.joinable())
                worker.join();
        }
    }

    PersistentHIPGraphLaunchWave(const PersistentHIPGraphLaunchWave &) = delete;
    PersistentHIPGraphLaunchWave &operator=(const PersistentHIPGraphLaunchWave &) = delete;

    /**
     * @brief Submit one graph on every device and wait until all events exist.
     *
     * Device execution remains asynchronous when this method returns. The
     * caller next waits on the endpoint events, preserving the production
     * all-submit-before-any-completion contract.
     */
    void submitAll()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        completed_count_ = 0;
        ++generation_;
        dispatch_cv_.notify_all();
        completion_cv_.wait(
            lock,
            [this]()
            {
                return completed_count_ == endpoints_.size();
            });
    }

private:
    /** @brief Own one HIP context and service every published launch generation. */
    void workerLoop(size_t index)
    {
        auto &endpoint = endpoints_[index];
        endpoint.worker_error = hipSetDevice(endpoint.device);
        uint64_t observed_generation = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++ready_count_;
        }
        ready_cv_.notify_one();

        while (true)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                dispatch_cv_.wait(
                    lock,
                    [this, observed_generation]()
                    {
                        return shutdown_ || generation_ > observed_generation;
                    });
                if (shutdown_)
                    return;
                observed_generation = generation_;
            }

            if (endpoint.worker_error == hipSuccess)
                endpoint.worker_error = hipGraphLaunch(
                    endpoint.executable,
                    endpoint.stream);
            if (endpoint.worker_error == hipSuccess)
                endpoint.worker_error = hipEventRecord(
                    endpoint.completion,
                    endpoint.stream);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completed_count_;
            }
            completion_cv_.notify_one();
        }
    }

    std::vector<RetainedDeviceGraph> &endpoints_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::condition_variable dispatch_cv_;
    std::condition_variable completion_cv_;
    size_t ready_count_ = 0;
    size_t completed_count_ = 0;
    uint64_t generation_ = 0;
    bool shutdown_ = false;
};

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

// ============================================================================
// TEST 6: Four-device retained endpoint graph submission topology
// ============================================================================

TEST(Perf__HIPGraphLaunchWave, FourDeviceSerialVsPersistentFanout)
{
    constexpr int kRequiredDevices = 4;
    constexpr int kEndpointGraphNodes = 19;
    constexpr int kWarmupIterations = 30;
    constexpr int kBenchmarkIterations = 1000;

    int available_devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&available_devices), hipSuccess);
    if (available_devices < kRequiredDevices)
    {
        GTEST_SKIP() << "Four-device endpoint launch proof requires four visible ROCm devices; found "
                     << available_devices;
    }

    std::vector<RetainedDeviceGraph> endpoints(kRequiredDevices);
    const auto destroyEndpoints = [&]()
    {
        for (auto &endpoint : endpoints)
            destroyRetainedDeviceGraph(endpoint);
    };
    for (int device = 0; device < kRequiredDevices; ++device)
    {
        const hipError_t status = initializeRetainedDeviceGraph(
            endpoints[static_cast<size_t>(device)],
            device,
            kEndpointGraphNodes);
        if (status != hipSuccess)
        {
            destroyEndpoints();
            FAIL() << "Could not prepare retained endpoint graph on ROCm device "
                   << device << ": " << hipGetErrorString(status);
            return;
        }
    }

    const auto waitForAllEvents = [&]() -> hipError_t
    {
        for (auto &endpoint : endpoints)
        {
            hipError_t status = hipSetDevice(endpoint.device);
            if (status != hipSuccess)
                return status;
            status = hipEventSynchronize(endpoint.completion);
            if (status != hipSuccess)
                return status;
        }
        return hipSuccess;
    };

    const auto submitSerial = [&]() -> hipError_t
    {
        for (auto &endpoint : endpoints)
        {
            hipError_t status = hipSetDevice(endpoint.device);
            if (status != hipSuccess)
                return status;
            status = hipGraphLaunch(endpoint.executable, endpoint.stream);
            if (status != hipSuccess)
                return status;
            status = hipEventRecord(endpoint.completion, endpoint.stream);
            if (status != hipSuccess)
                return status;
        }
        return hipSuccess;
    };

    std::vector<double> serial_submission_samples;
    std::vector<double> serial_service_samples;
    std::vector<double> fanout_submission_samples;
    std::vector<double> fanout_service_samples;
    serial_submission_samples.reserve(kBenchmarkIterations);
    serial_service_samples.reserve(kBenchmarkIterations);
    fanout_submission_samples.reserve(kBenchmarkIterations);
    fanout_service_samples.reserve(kBenchmarkIterations);

    hipError_t benchmark_error = hipSuccess;
    {
        PersistentHIPGraphLaunchWave fanout(endpoints);

        // Warm both paths before timing so graph instantiation, worker startup,
        // and initial HIP context work cannot bias either implementation.
        for (int iteration = 0; iteration < kWarmupIterations; ++iteration)
        {
            benchmark_error = submitSerial();
            if (benchmark_error != hipSuccess)
                break;
            benchmark_error = waitForAllEvents();
            if (benchmark_error != hipSuccess)
                break;
            fanout.submitAll();
            benchmark_error = waitForAllEvents();
            if (benchmark_error != hipSuccess)
                break;
        }

        for (int iteration = 0;
             benchmark_error == hipSuccess &&
             iteration < kBenchmarkIterations;
             ++iteration)
        {
            const auto serial_start = std::chrono::steady_clock::now();
            benchmark_error = submitSerial();
            const auto serial_submitted = std::chrono::steady_clock::now();
            if (benchmark_error != hipSuccess)
                break;
            benchmark_error = waitForAllEvents();
            const auto serial_completed = std::chrono::steady_clock::now();
            if (benchmark_error != hipSuccess)
                break;
            serial_submission_samples.push_back(
                std::chrono::duration<double, std::micro>(
                    serial_submitted - serial_start)
                    .count());
            serial_service_samples.push_back(
                std::chrono::duration<double, std::micro>(
                    serial_completed - serial_start)
                    .count());

            const auto fanout_start = std::chrono::steady_clock::now();
            fanout.submitAll();
            const auto fanout_submitted = std::chrono::steady_clock::now();
            benchmark_error = waitForAllEvents();
            const auto fanout_completed = std::chrono::steady_clock::now();
            if (benchmark_error != hipSuccess)
                break;
            fanout_submission_samples.push_back(
                std::chrono::duration<double, std::micro>(
                    fanout_submitted - fanout_start)
                    .count());
            fanout_service_samples.push_back(
                std::chrono::duration<double, std::micro>(
                    fanout_completed - fanout_start)
                    .count());
        }

        for (const auto &endpoint : endpoints)
        {
            if (benchmark_error == hipSuccess &&
                endpoint.worker_error != hipSuccess)
            {
                benchmark_error = endpoint.worker_error;
            }
        }
    }

    if (benchmark_error != hipSuccess)
    {
        const std::string error = hipGetErrorString(benchmark_error);
        destroyEndpoints();
        FAIL() << "Four-device endpoint launch benchmark failed: " << error;
        return;
    }

    auto serial_submission = computeStats(
        serial_submission_samples,
        kRequiredDevices);
    auto serial_service = computeStats(
        serial_service_samples,
        kRequiredDevices);
    auto fanout_submission = computeStats(
        fanout_submission_samples,
        kRequiredDevices);
    auto fanout_service = computeStats(
        fanout_service_samples,
        kRequiredDevices);

    printf("\n");
    printf("Four-device retained HIP endpoint graph (%d nodes/device, median of %d):\n",
           kEndpointGraphNodes,
           kBenchmarkIterations);
    printf("  serial coordinator: submission=%7.1f us, all-complete=%7.1f us\n",
           serial_submission.median_us,
           serial_service.median_us);
    printf("  persistent fan-out: submission=%7.1f us, all-complete=%7.1f us\n",
           fanout_submission.median_us,
           fanout_service.median_us);
    printf("  service speedup:    %7.2fx\n",
           serial_service.median_us / fanout_service.median_us);

    EXPECT_EQ(serial_service.iterations, kBenchmarkIterations);
    EXPECT_EQ(fanout_service.iterations, kBenchmarkIterations);
    EXPECT_GT(serial_service.median_us, 0.0);
    EXPECT_GT(fanout_service.median_us, 0.0);
    destroyEndpoints();
}

// ============================================================================
// TEST 7: Four-device retained-family graph instantiation contention
// ============================================================================

TEST(Perf__HIPGraphInstantiationWave, FourDeviceConcurrentVsSerialPublication)
{
    constexpr int kRequiredDevices = 4;
    constexpr int kGraphNodes = 80;
    constexpr int kExecutablesPerDevice = 32;

    int available_devices = 0;
    ASSERT_EQ(hipGetDeviceCount(&available_devices), hipSuccess);
    if (available_devices < kRequiredDevices)
    {
        GTEST_SKIP()
            << "Four-device graph-instantiation proof requires four visible "
               "ROCm devices; found "
            << available_devices;
    }

    std::vector<CapturedInstantiationTemplate> templates(
        kRequiredDevices);
    const auto destroyTemplates = [&]()
    {
        for (auto &graph : templates)
            destroyCapturedInstantiationTemplate(graph);
    };
    for (int device = 0; device < kRequiredDevices; ++device)
    {
        const hipError_t status = captureInstantiationTemplate(
            templates[static_cast<std::size_t>(device)],
            device,
            kGraphNodes);
        if (status != hipSuccess)
        {
            destroyTemplates();
            FAIL() << "Could not capture graph-instantiation template on "
                      "ROCm device "
                   << device << ": " << hipGetErrorString(status);
            return;
        }
    }

    /* Run the contended form first.  Destroying every executable before the
     * serial wave keeps retained-family cardinality identical; the serial
     * result then shows whether publication order, rather than graph topology,
     * owns the scaling cliff. */
    const InstantiationWaveResult concurrent = instantiateRetainedFamily(
        templates,
        kExecutablesPerDevice,
        /*concurrent=*/true);
    const InstantiationWaveResult serial = instantiateRetainedFamily(
        templates,
        kExecutablesPerDevice,
        /*concurrent=*/false);
    destroyTemplates();

    ASSERT_EQ(concurrent.status, hipSuccess)
        << "Concurrent retained-family publication failed: "
        << hipGetErrorString(concurrent.status);
    ASSERT_EQ(serial.status, hipSuccess)
        << "Serial retained-family publication failed: "
        << hipGetErrorString(serial.status);

    std::printf(
        "\nHIP retained-family instantiation: devices=%d nodes=%d "
        "executables/device=%d concurrent=%.3f ms serial=%.3f ms "
        "concurrent/serial=%.3fx\n",
        kRequiredDevices,
        kGraphNodes,
        kExecutablesPerDevice,
        concurrent.elapsed_ms,
        serial.elapsed_ms,
        serial.elapsed_ms > 0.0
            ? concurrent.elapsed_ms / serial.elapsed_ms
            : 0.0);
}

} // anonymous namespace

#else // !HAVE_ROCM

TEST(Perf__HIPGraphLaunchOverhead, SkipNoROCm)
{
    GTEST_SKIP() << "ROCm not available";
}

#endif // HAVE_ROCM
