/**
 * @file Perf__ROCmMoERouterSoftmaxTopK.cpp
 * @brief ROCm speedometer for batch-invariant MoE router softmax and top-k.
 *
 * This is the gfx906 counterpart to the CUDA router speedometer.  It isolates
 * the graph-captured softmax/top-k kernel at Qwen3.6's 256-expert, top-8
 * geometry and compares exact M=1583 with the production M=2048 graph bucket.
 * Correctness is owned by the integration suite; this test guards the economic
 * contract and supplies a compact launch for `rocprof` VGPR, occupancy, LDS,
 * scratch-spill, and throughput analysis.
 *
 * Device memory is allocated and populated before capture.  The timed region
 * contains graph launches only, followed by one HIP event observation after
 * all iterations have been enqueued.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_ROCM
extern "C" bool hipMoE_softmax_topk(
    float *logits,
    int *expert_indices,
    float *expert_weights,
    int seq_len,
    int num_experts,
    int top_k,
    bool normalize_weights,
    int device_idx,
    void *stream,
    const int *device_effective_seq_len);
#endif

namespace
{
#ifdef HAVE_ROCM
    /**
     * @brief Own one setup-time HIP allocation used by the perf fixture.
     */
    class HipPerfBuffer
    {
    public:
        explicit HipPerfBuffer(size_t bytes)
        {
            const hipError_t status = hipMalloc(&pointer_, bytes);
            if (status != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("hipMalloc failed for MoE router perf buffer: ") +
                    hipGetErrorString(status));
            }
        }

        ~HipPerfBuffer()
        {
            if (pointer_)
                (void)hipFree(pointer_);
        }

        HipPerfBuffer(const HipPerfBuffer &) = delete;
        HipPerfBuffer &operator=(const HipPerfBuffer &) = delete;

        void *get() const { return pointer_; }

    private:
        void *pointer_ = nullptr;
    };

    /**
     * @brief Own one captured HIP graph and its executable instance.
     */
    class HipRouterGraph
    {
    public:
        ~HipRouterGraph()
        {
            if (executable_)
                (void)hipGraphExecDestroy(executable_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
        }

        hipGraph_t *graphAddress() { return &graph_; }
        hipGraphExec_t *executableAddress() { return &executable_; }
        hipGraphExec_t executable() const { return executable_; }

    private:
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t executable_ = nullptr;
    };

    int routerPerfIterations()
    {
        constexpr int kDefaultIterations = 1000;
        const char *raw = std::getenv("LLAMINAR_MOE_ROUTER_PERF_ITERS");
        if (!raw || !*raw)
            return kDefaultIterations;
        return std::max(1, std::atoi(raw));
    }

    /**
     * @brief Capture and time one exact or padded production router launch.
     *
     * @param launch_rows Number of row work-groups captured in the graph.
     * @param active_rows Number of semantically valid rows in the bucket.
     * @param average_us Receives average graph replay duration in microseconds.
     */
    void benchmarkHipRouterSoftmaxTopK(
        int launch_rows,
        int active_rows,
        double *average_us)
    {
        constexpr int kNumExperts = 256;
        constexpr int kTopK = 8;
        constexpr int kWarmups = 8;

        ASSERT_GT(launch_rows, 0);
        ASSERT_GT(active_rows, 0);
        ASSERT_LE(active_rows, launch_rows);
        ASSERT_NE(average_us, nullptr);
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);

        const size_t logits_count =
            static_cast<size_t>(launch_rows) * kNumExperts;
        const size_t topk_count =
            static_cast<size_t>(launch_rows) * kTopK;
        std::vector<float> host_logits(logits_count);
        for (size_t i = 0; i < host_logits.size(); ++i)
        {
            host_logits[i] =
                0.037f * std::sin(static_cast<float>(i % 65521) * 0.0073f) +
                0.019f * std::cos(static_cast<float>(i % 32749) * 0.011f);
        }

        HipPerfBuffer logits(logits_count * sizeof(float));
        HipPerfBuffer indices(topk_count * sizeof(int));
        HipPerfBuffer weights(topk_count * sizeof(float));
        HipPerfBuffer effective_rows(sizeof(int));
        ASSERT_EQ(
            hipMemcpyAsync(
                logits.get(),
                host_logits.data(),
                logits_count * sizeof(float),
                hipMemcpyHostToDevice,
                stream),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(
                effective_rows.get(),
                &active_rows,
                sizeof(active_rows),
                hipMemcpyHostToDevice,
                stream),
            hipSuccess);

        const int *device_effective_rows =
            active_rows == launch_rows
                ? nullptr
                : static_cast<const int *>(effective_rows.get());
        auto launch = [&]()
        {
            return hipMoE_softmax_topk(
                static_cast<float *>(logits.get()),
                static_cast<int *>(indices.get()),
                static_cast<float *>(weights.get()),
                launch_rows,
                kNumExperts,
                kTopK,
                /*normalize_weights=*/true,
                /*device_idx=*/0,
                stream,
                device_effective_rows);
        };

        ASSERT_TRUE(launch());
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipRouterGraph graph;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        const bool captured = launch();
        const hipError_t capture_status =
            hipStreamEndCapture(stream, graph.graphAddress());
        ASSERT_TRUE(captured);
        ASSERT_EQ(capture_status, hipSuccess)
            << hipGetErrorString(capture_status);
        ASSERT_NE(*graph.graphAddress(), nullptr);
        ASSERT_EQ(
            hipGraphInstantiate(
                graph.executableAddress(),
                *graph.graphAddress(),
                nullptr,
                nullptr,
                0),
            hipSuccess);

        for (int warmup = 0; warmup < kWarmups; ++warmup)
            ASSERT_EQ(hipGraphLaunch(graph.executable(), stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        ASSERT_EQ(hipEventCreate(&start), hipSuccess);
        ASSERT_EQ(hipEventCreate(&stop), hipSuccess);
        ASSERT_EQ(hipEventRecord(start, stream), hipSuccess);
        const int iterations = routerPerfIterations();
        for (int iteration = 0; iteration < iterations; ++iteration)
            ASSERT_EQ(hipGraphLaunch(graph.executable(), stream), hipSuccess);
        ASSERT_EQ(hipEventRecord(stop, stream), hipSuccess);
        ASSERT_EQ(hipEventSynchronize(stop), hipSuccess);

        float elapsed_ms = 0.0f;
        ASSERT_EQ(
            hipEventElapsedTime(&elapsed_ms, start, stop),
            hipSuccess);
        ASSERT_EQ(hipEventDestroy(start), hipSuccess);
        ASSERT_EQ(hipEventDestroy(stop), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        *average_us =
            static_cast<double>(elapsed_ms) * 1000.0 /
            static_cast<double>(iterations);
    }
#endif
}

TEST(Perf__MoERouterSoftmaxTopK, ROCm_Qwen36ExactAndPaddedGraphReplay)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";

    constexpr int kActiveRows = 1583;
    constexpr int kBucketRows = 2048;
    constexpr int kNumExperts = 256;
    double exact_us = 0.0;
    double padded_us = 0.0;
    benchmarkHipRouterSoftmaxTopK(
        kActiveRows, kActiveRows, &exact_us);
    benchmarkHipRouterSoftmaxTopK(
        kBucketRows, kActiveRows, &padded_us);
    ASSERT_GT(exact_us, 0.0);
    ASSERT_GT(padded_us, 0.0);

    const double active_values_per_us =
        static_cast<double>(kActiveRows) * kNumExperts / padded_us;
    std::cout << std::fixed << std::setprecision(3)
              << "backend,active_rows,bucket_rows,exact_us,padded_us,"
                 "padding_ratio,active_expert_values_per_us\n"
              << "rocm," << kActiveRows << ',' << kBucketRows << ','
              << exact_us << ',' << padded_us << ','
              << padded_us / exact_us << ','
              << active_values_per_us << '\n';

    EXPECT_LT(padded_us, exact_us * 1.60);
#endif
}
