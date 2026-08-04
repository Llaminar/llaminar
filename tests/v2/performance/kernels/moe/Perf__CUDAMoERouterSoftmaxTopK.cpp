/**
 * @file Perf__CUDAMoERouterSoftmaxTopK.cpp
 * @brief CUDA speedometer for batch-invariant MoE router softmax and top-k.
 *
 * The Qwen3.6 MoE router owns one row per CUDA block and must produce the same
 * active-row bytes whether prefill runs at its exact token count or through a
 * larger graph bucket.  The corresponding integration regression proves that
 * numerical contract.  This file measures the same production geometry in
 * isolation so a correctness fix cannot silently serialize the row, spill
 * registers, or make padded graph replay uneconomical.
 *
 * All device allocations and input uploads occur before graph capture.  The
 * timed region contains only graph launches.  CUDA events provide the final
 * host-visible timing boundary; there are no allocations, transfers, or device
 * synchronizations in the measured hot path.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
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

#ifdef HAVE_CUDA
extern "C" bool cudaMoE_softmax_topk(
    float *logits,
    float *expert_indices,
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
#ifdef HAVE_CUDA
    /**
     * @brief Own one setup-time CUDA allocation used by the perf fixture.
     *
     * Production obtains these buffers from its persistent workspace.  The
     * standalone speedometer has no graph arena, so this small owner mirrors
     * that lifetime: allocate once before capture and release after every
     * replay and timing event has completed.
     */
    class CudaPerfBuffer
    {
    public:
        explicit CudaPerfBuffer(size_t bytes)
        {
            const cudaError_t status = cudaMalloc(&pointer_, bytes);
            if (status != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("cudaMalloc failed for MoE router perf buffer: ") +
                    cudaGetErrorString(status));
            }
        }

        ~CudaPerfBuffer()
        {
            if (pointer_)
                (void)cudaFree(pointer_);
        }

        CudaPerfBuffer(const CudaPerfBuffer &) = delete;
        CudaPerfBuffer &operator=(const CudaPerfBuffer &) = delete;

        void *get() const { return pointer_; }

    private:
        void *pointer_ = nullptr;
    };

    /**
     * @brief Own one captured CUDA graph and its executable instance.
     */
    class CudaRouterGraph
    {
    public:
        ~CudaRouterGraph()
        {
            if (executable_)
                (void)cudaGraphExecDestroy(executable_);
            if (graph_)
                (void)cudaGraphDestroy(graph_);
        }

        cudaGraph_t *graphAddress() { return &graph_; }
        cudaGraphExec_t *executableAddress() { return &executable_; }
        cudaGraphExec_t executable() const { return executable_; }

    private:
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t executable_ = nullptr;
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
     * @param launch_rows Number of row blocks captured in the graph.
     * @param active_rows Number of semantically valid rows in the bucket.
     * @param average_us Receives average graph replay duration in microseconds.
     */
    void benchmarkCudaRouterSoftmaxTopK(
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
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);

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

        CudaPerfBuffer logits(logits_count * sizeof(float));
        CudaPerfBuffer indices(topk_count * sizeof(float));
        CudaPerfBuffer weights(topk_count * sizeof(float));
        CudaPerfBuffer effective_rows(sizeof(int));
        ASSERT_EQ(
            cudaMemcpyAsync(
                logits.get(),
                host_logits.data(),
                logits_count * sizeof(float),
                cudaMemcpyHostToDevice,
                stream),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                effective_rows.get(),
                &active_rows,
                sizeof(active_rows),
                cudaMemcpyHostToDevice,
                stream),
            cudaSuccess);

        const int *device_effective_rows =
            active_rows == launch_rows
                ? nullptr
                : static_cast<const int *>(effective_rows.get());
        auto launch = [&]()
        {
            return cudaMoE_softmax_topk(
                static_cast<float *>(logits.get()),
                static_cast<float *>(indices.get()),
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
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        CudaRouterGraph graph;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        const bool captured = launch();
        const cudaError_t capture_status =
            cudaStreamEndCapture(stream, graph.graphAddress());
        ASSERT_TRUE(captured);
        ASSERT_EQ(capture_status, cudaSuccess)
            << cudaGetErrorString(capture_status);
        ASSERT_NE(*graph.graphAddress(), nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(
                graph.executableAddress(),
                *graph.graphAddress(),
                nullptr,
                nullptr,
                0),
            cudaSuccess);

        for (int warmup = 0; warmup < kWarmups; ++warmup)
            ASSERT_EQ(cudaGraphLaunch(graph.executable(), stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(start, stream), cudaSuccess);
        const int iterations = routerPerfIterations();
        for (int iteration = 0; iteration < iterations; ++iteration)
            ASSERT_EQ(cudaGraphLaunch(graph.executable(), stream), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float elapsed_ms = 0.0f;
        ASSERT_EQ(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        *average_us =
            static_cast<double>(elapsed_ms) * 1000.0 /
            static_cast<double>(iterations);
    }
#endif
}

TEST(Perf__MoERouterSoftmaxTopK, CUDA_Qwen36ExactAndPaddedGraphReplay)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";

    constexpr int kActiveRows = 1583;
    constexpr int kBucketRows = 2048;
    constexpr int kNumExperts = 256;
    double exact_us = 0.0;
    double padded_us = 0.0;
    benchmarkCudaRouterSoftmaxTopK(
        kActiveRows, kActiveRows, &exact_us);
    benchmarkCudaRouterSoftmaxTopK(
        kBucketRows, kActiveRows, &padded_us);
    ASSERT_GT(exact_us, 0.0);
    ASSERT_GT(padded_us, 0.0);

    const double active_values_per_us =
        static_cast<double>(kActiveRows) * kNumExperts / padded_us;
    std::cout << std::fixed << std::setprecision(3)
              << "backend,active_rows,bucket_rows,exact_us,padded_us,"
                 "padding_ratio,active_expert_values_per_us\n"
              << "cuda," << kActiveRows << ',' << kBucketRows << ','
              << exact_us << ',' << padded_us << ','
              << padded_us / exact_us << ','
              << active_values_per_us << '\n';

    /*
     * Inactive bucket rows perform only the explicit output clearing required
     * by the graph contract.  A 29% larger launch should not double latency.
     * The broad bound rejects accidental row serialization while tolerating
     * normal clock and display-load variance on development GPUs.
     */
    EXPECT_LT(padded_us, exact_us * 1.60);
#endif
}
