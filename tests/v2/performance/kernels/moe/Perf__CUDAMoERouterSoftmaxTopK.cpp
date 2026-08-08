/**
 * @file Perf__CUDAMoERouterSoftmaxTopK.cpp
 * @brief CUDA speedometers for MoE router logits, softmax, and top-k.
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

#include "kernels/cuda/moe/CUDAMoERouterPrefillPolicy.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
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

    /** @return Timed graph replays for one router-logit candidate. */
    int routerLogitsPerfIterations()
    {
        constexpr int kDefaultIterations = 500;
        const char *raw =
            std::getenv("LLAMINAR_MOE_ROUTER_LOGITS_PERF_ITERS");
        if (!raw || !*raw)
            return kDefaultIterations;
        return std::max(1, std::atoi(raw));
    }

    /**
     * @brief Capture, time, and materialize one exact router-logit geometry.
     *
     * Input allocation and upload are owned by the caller.  This function
     * performs one untimed setup launch, captures the production bridge, and
     * uses CUDA events around graph replay only.  The single D2H copy occurs
     * after timing and exists solely for byte-equivalence certification.
     */
    double benchmarkCudaRouteLogitsGeometry(
        const float *device_hidden,
        const float *device_gate,
        float *device_logits,
        int seq_len,
        int d_model,
        int num_experts,
        llaminar2::CUDAMoERouterPrefillGeometry geometry,
        cudaStream_t stream,
        std::vector<float> *host_logits)
    {
        constexpr int kWarmups = 8;
        if (!device_hidden || !device_gate || !device_logits || !stream ||
            !host_logits)
        {
            throw std::invalid_argument(
                "router-logit benchmark requires persistent device buffers, "
                "an explicit stream, and output storage");
        }

        auto launch = [&]()
        {
            return cudaMoE_route_logits_with_geometry(
                device_hidden,
                device_gate,
                device_logits,
                seq_len,
                d_model,
                num_experts,
                /*device_idx=*/0,
                stream,
                geometry);
        };

        EXPECT_TRUE(launch());
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        CudaRouterGraph graph;
        EXPECT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        const bool captured = launch();
        const cudaError_t capture_status =
            cudaStreamEndCapture(stream, graph.graphAddress());
        EXPECT_TRUE(captured);
        EXPECT_EQ(capture_status, cudaSuccess)
            << cudaGetErrorString(capture_status);
        EXPECT_NE(*graph.graphAddress(), nullptr);
        EXPECT_EQ(
            cudaGraphInstantiate(
                graph.executableAddress(),
                *graph.graphAddress(),
                nullptr,
                nullptr,
                0),
            cudaSuccess);

        for (int warmup = 0; warmup < kWarmups; ++warmup)
            EXPECT_EQ(cudaGraphLaunch(graph.executable(), stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
        EXPECT_EQ(cudaEventCreate(&stop), cudaSuccess);
        EXPECT_EQ(cudaEventRecord(start, stream), cudaSuccess);
        const int iterations = routerLogitsPerfIterations();
        for (int iteration = 0; iteration < iterations; ++iteration)
            EXPECT_EQ(cudaGraphLaunch(graph.executable(), stream), cudaSuccess);
        EXPECT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float elapsed_ms = 0.0f;
        EXPECT_EQ(cudaEventElapsedTime(&elapsed_ms, start, stop), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(start), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(stop), cudaSuccess);

        host_logits->resize(
            static_cast<size_t>(seq_len) *
            static_cast<size_t>(num_experts));
        EXPECT_EQ(
            cudaMemcpyAsync(
                host_logits->data(),
                device_logits,
                host_logits->size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream),
            cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        return static_cast<double>(elapsed_ms) * 1000.0 /
            static_cast<double>(iterations);
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

/**
 * @brief Tournament CUDA FP32 router tiles across Qwen3.6 graph buckets.
 *
 * The production graph computes route logits over the complete stable bucket,
 * not merely the active prompt rows.  This sweep therefore covers every common
 * bucket through 1024 at K=2048, E=256.  Promotion requires exact bytes, graph
 * capture, zero compiler-local storage, and a measured replay win over the
 * installed tile at each bucket.
 */
TEST(Perf__MoERouterLogits, CUDA_Qwen36BucketGeometryTournament)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "No CUDA device available";

    constexpr int kMaxSeqLen = 4096;
    constexpr int kDModel = 2048;
    constexpr int kNumExperts = 256;
    struct BucketWinner
    {
        int seq_len;
        llaminar2::CUDAMoERouterPrefillGeometry geometry;
    };
    using Geometry = llaminar2::CUDAMoERouterPrefillGeometry;
    constexpr std::array kBuckets{
        BucketWinner{64, Geometry::Tile16x16},
        BucketWinner{128, Geometry::Tile32x16},
        BucketWinner{256, Geometry::Tile32x32},
        BucketWinner{384, Geometry::Tile64x32},
        BucketWinner{512, Geometry::Tile32x32K16},
        BucketWinner{544, Geometry::Tile32x32K16},
        BucketWinner{576, Geometry::Tile32x32K16},
        BucketWinner{600, Geometry::Tile32x32K16},
        BucketWinner{608, Geometry::Tile32x32K16},
        BucketWinner{640, Geometry::Tile32x32K16},
        BucketWinner{672, Geometry::Tile64x32},
        BucketWinner{704, Geometry::Tile64x32},
        BucketWinner{736, Geometry::Tile64x32},
        BucketWinner{768, Geometry::Tile64x32},
        BucketWinner{1024, Geometry::Tile64x32},
        BucketWinner{1280, Geometry::Tile64x32},
        BucketWinner{1536, Geometry::Tile64x32},
        BucketWinner{2048, Geometry::Tile64x64},
        BucketWinner{2560, Geometry::Tile64x64},
        BucketWinner{3072, Geometry::Tile64x64},
        BucketWinner{4096, Geometry::Tile64x64},
    };
    constexpr std::array kGeometries{
        llaminar2::CUDAMoERouterPrefillGeometry::Tile64x64,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile64x32,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile32x64,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile32x32,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile32x32K16,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile32x24,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile24x32,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile64x16,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile32x16,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile16x64,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile16x32,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile16x16,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile8x64,
        llaminar2::CUDAMoERouterPrefillGeometry::Tile8x32,
    };

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        cudaSuccess);

    std::vector<float> host_hidden(
        static_cast<size_t>(kMaxSeqLen) * kDModel);
    std::vector<float> host_gate(
        static_cast<size_t>(kNumExperts) * kDModel);
    for (size_t index = 0; index < host_hidden.size(); ++index)
    {
        host_hidden[index] =
            0.031f * std::sin(static_cast<float>(index % 65521) * 0.0073f) +
            0.013f * std::cos(static_cast<float>(index % 32749) * 0.011f);
    }
    for (size_t index = 0; index < host_gate.size(); ++index)
    {
        host_gate[index] =
            0.021f * std::cos(static_cast<float>(index % 16381) * 0.017f) -
            0.009f * std::sin(static_cast<float>(index % 8191) * 0.023f);
    }

    CudaPerfBuffer device_hidden(host_hidden.size() * sizeof(float));
    CudaPerfBuffer device_gate(host_gate.size() * sizeof(float));
    CudaPerfBuffer device_logits(
        static_cast<size_t>(kMaxSeqLen) * kNumExperts * sizeof(float));
    ASSERT_EQ(
        cudaMemcpyAsync(
            device_hidden.get(),
            host_hidden.data(),
            host_hidden.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            device_gate.get(),
            host_gate.data(),
            host_gate.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::cout
        << "seq_len,geometry,tile_m,tile_n,grid_blocks,registers,local_bytes,"
           "shared_bytes,blocks_per_sm,replay_us,relative_to_64x64\n";
    for (const BucketWinner &bucket : kBuckets)
    {
        const int seq_len = bucket.seq_len;
        ASSERT_EQ(
            llaminar2::selectCUDAMoERouterPrefillGeometry(
                seq_len, kDModel, kNumExperts),
            bucket.geometry)
            << "capture-time selector drifted from the certified M="
            << seq_len << " winner";
        std::vector<float> reference_logits;
        double reference_us = 0.0;
        for (size_t candidate_index = 0;
             candidate_index < kGeometries.size();
             ++candidate_index)
        {
            const auto geometry = kGeometries[candidate_index];
            const auto spec =
                llaminar2::cudaMoERouterPrefillGeometrySpec(geometry);
            int registers_per_thread = 0;
            size_t local_memory_bytes_per_thread = 0;
            size_t static_shared_memory_bytes = 0;
            int max_threads_per_block = 0;
            int max_active_blocks_per_sm = 0;
            ASSERT_TRUE(cudaMoE_route_logits_query_geometry_resources(
                geometry,
                &registers_per_thread,
                &local_memory_bytes_per_thread,
                &static_shared_memory_bytes,
                &max_threads_per_block,
                &max_active_blocks_per_sm));
            ASSERT_EQ(local_memory_bytes_per_thread, 0u)
                << spec.name << " spills or owns compiler-local storage";
            ASSERT_GE(max_threads_per_block, spec.threads_per_block);
            ASSERT_GT(max_active_blocks_per_sm, 0);

            std::vector<float> candidate_logits;
            const double replay_us = benchmarkCudaRouteLogitsGeometry(
                static_cast<const float *>(device_hidden.get()),
                static_cast<const float *>(device_gate.get()),
                static_cast<float *>(device_logits.get()),
                seq_len,
                kDModel,
                kNumExperts,
                geometry,
                stream,
                &candidate_logits);
            ASSERT_GT(replay_us, 0.0);

            if (candidate_index == 0)
            {
                reference_logits = candidate_logits;
                reference_us = replay_us;
            }
            else
            {
                ASSERT_EQ(candidate_logits.size(), reference_logits.size());
                ASSERT_EQ(
                    std::memcmp(
                        candidate_logits.data(),
                        reference_logits.data(),
                        reference_logits.size() * sizeof(float)),
                    0)
                    << "M=" << seq_len << ' ' << spec.name
                    << " changed router-logit bytes relative to production 64x64";
            }

            const int grid_blocks =
                ((seq_len + spec.tile_rows - 1) / spec.tile_rows) *
                ((kNumExperts + spec.tile_experts - 1) /
                 spec.tile_experts);
            std::cout << std::fixed << std::setprecision(3)
                      << seq_len << ','
                      << spec.name << ','
                      << spec.tile_rows << ','
                      << spec.tile_experts << ','
                      << grid_blocks << ','
                      << registers_per_thread << ','
                      << local_memory_bytes_per_thread << ','
                      << static_shared_memory_bytes << ','
                      << max_active_blocks_per_sm << ','
                      << replay_us << ','
                      << (replay_us / reference_us) << '\n';
        }
    }

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}
