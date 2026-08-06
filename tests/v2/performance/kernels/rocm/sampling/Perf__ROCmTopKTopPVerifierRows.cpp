/**
 * @file Perf__ROCmTopKTopPVerifierRows.cpp
 * @brief ROCm speedometer for graph-captured stochastic-verifier Top-K/Top-P.
 *
 * The stochastic MTP verifier converts every active LM-head row into a compact
 * Top-K/Top-P distribution before rejection sampling.  Qwen 3.6 uses a
 * 248,320-token vocabulary and Top-K 40, so an inefficient vocabulary scan is
 * paid once per verifier transaction even though the resulting distribution
 * is tiny.  This fixture times the exact production two-kernel pipeline across
 * M=1,2,4,8,16 and keeps the device-owned active-row scalar in the launch
 * contract used by captured inference.
 *
 * All allocation and input publication happen before graph capture.  The timed
 * region consists exclusively of graph replay on one explicit non-blocking HIP
 * stream, followed by one event observation after every iteration has been
 * enqueued.  Correctness and serial-row byte equivalence are owned by
 * Test__GPUSamplingKernels; this executable is deliberately a narrow economy
 * and profiler target for launch geometry, VGPR/SGPR pressure, LDS occupancy,
 * scratch spills, and vocabulary throughput.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_ROCM
extern "C" bool rocmOps_topk_topp_distributions_f32(
    const float *data,
    int row_count,
    int n,
    int row_stride,
    int k,
    float top_p,
    float temperature,
    int *out_token_ids,
    int out_stride,
    float *out_probs,
    float *scratch_values,
    int *scratch_indices,
    int scratch_capacity,
    const int *active_rows,
    int device_idx,
    void *stream);
#endif

namespace
{
#ifdef HAVE_ROCM
    constexpr int kVocabSize = 248320;
    constexpr int kTopK = 40;
    constexpr int kMaximumRows = 16;
    constexpr int kMaximumPartialBlocks = 128;

    /**
     * @brief Own one setup-time HIP allocation used by the speedometer.
     *
     * Destruction occurs only after all timed work has completed.  The class is
     * intentionally unavailable to the captured launch lambda so introducing a
     * hot-path allocation requires an explicit structural change to the test.
     */
    class HipPerfBuffer
    {
    public:
        /**
         * @brief Allocate a persistent device buffer before graph capture.
         * @param bytes Required allocation size in bytes.
         */
        explicit HipPerfBuffer(size_t bytes)
        {
            const hipError_t status = hipMalloc(&pointer_, bytes);
            if (status != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("hipMalloc failed for sampling perf buffer: ") +
                    hipGetErrorString(status));
            }
        }

        /** @brief Release the setup-time allocation after timing completes. */
        ~HipPerfBuffer()
        {
            if (pointer_)
                (void)hipFree(pointer_);
        }

        HipPerfBuffer(const HipPerfBuffer &) = delete;
        HipPerfBuffer &operator=(const HipPerfBuffer &) = delete;

        /** @brief Return the stable device pointer embedded in the graph. */
        void *get() const { return pointer_; }

    private:
        void *pointer_ = nullptr;
    };

    /**
     * @brief Own one captured graph and its executable instance.
     */
    class HipSamplingGraph
    {
    public:
        /** @brief Destroy graph resources after all replays have completed. */
        ~HipSamplingGraph()
        {
            if (executable_)
                (void)hipGraphExecDestroy(executable_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
        }

        HipSamplingGraph(const HipSamplingGraph &) = delete;
        HipSamplingGraph &operator=(const HipSamplingGraph &) = delete;
        HipSamplingGraph() = default;

        /** @brief Return storage for hipStreamEndCapture(). */
        hipGraph_t *graphAddress() { return &graph_; }

        /** @brief Return storage for hipGraphInstantiate(). */
        hipGraphExec_t *executableAddress() { return &executable_; }

        /** @brief Return the immutable executable used by timed replays. */
        hipGraphExec_t executable() const { return executable_; }

    private:
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t executable_ = nullptr;
    };

    /**
     * @brief Return the requested replay count, with a stable profiler override.
     */
    int samplingPerfIterations()
    {
        constexpr int kDefaultIterations = 500;
        const char *raw = std::getenv("LLAMINAR_ROCM_SAMPLING_PERF_ITERS");
        if (!raw || !*raw)
            return kDefaultIterations;
        return std::max(1, std::atoi(raw));
    }

    /**
     * @brief Return the setup-only warmup count used before event timing.
     *
     * A profiler launch may set this to zero so its dispatch range contains
     * only the one eager validation pair and the requested timed replay pair.
     */
    int samplingPerfWarmupIterations()
    {
        constexpr int kDefaultWarmupIterations = 12;
        const char *raw =
            std::getenv("LLAMINAR_ROCM_SAMPLING_PERF_WARMUPS");
        if (!raw || !*raw)
            return kDefaultWarmupIterations;
        return std::max(0, std::atoi(raw));
    }

    /**
     * @brief Capture and time the production stochastic-verifier distribution.
     *
     * @param row_count Captured verifier depth and active row count.
     * @param average_us Receives average two-kernel graph replay latency.
     */
    void benchmarkTopKTopPVerifierRows(
        int row_count,
        double *average_us)
    {
        ASSERT_GT(row_count, 0);
        ASSERT_LE(row_count, kMaximumRows);
        ASSERT_NE(average_us, nullptr);
        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        ASSERT_NE(stream, nullptr);

        const size_t logits_count =
            static_cast<size_t>(row_count) * kVocabSize;
        std::vector<float> host_logits(logits_count);
        for (int row = 0; row < row_count; ++row)
        {
            for (int token = 0; token < kVocabSize; ++token)
            {
                host_logits[
                    static_cast<size_t>(row) * kVocabSize + token] =
                    -18.0f -
                    0.00037f * static_cast<float>(
                        (token * 37 + row * 101) % 997);
            }
            for (int rank = 0; rank < kTopK; ++rank)
            {
                const int token =
                    (row * 15401 + rank * 7919 + 321) % kVocabSize;
                host_logits[
                    static_cast<size_t>(row) * kVocabSize + token] =
                    6.0f - 0.071f * static_cast<float>(rank / 2) +
                    0.003f * static_cast<float>(row);
            }
        }

        const int scratch_capacity =
            row_count * kMaximumPartialBlocks * kTopK;
        HipPerfBuffer logits(logits_count * sizeof(float));
        HipPerfBuffer token_ids(
            static_cast<size_t>(row_count) * kTopK * sizeof(int));
        HipPerfBuffer probabilities(
            static_cast<size_t>(row_count) * kTopK * sizeof(float));
        HipPerfBuffer scratch_values(
            static_cast<size_t>(scratch_capacity) * sizeof(float));
        HipPerfBuffer scratch_indices(
            static_cast<size_t>(scratch_capacity) * sizeof(int));
        HipPerfBuffer active_rows(sizeof(int));

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
                active_rows.get(),
                &row_count,
                sizeof(row_count),
                hipMemcpyHostToDevice,
                stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        auto launch = [&]()
        {
            return rocmOps_topk_topp_distributions_f32(
                static_cast<const float *>(logits.get()),
                row_count,
                kVocabSize,
                kVocabSize,
                kTopK,
                /*top_p=*/0.9f,
                /*temperature=*/0.7f,
                static_cast<int *>(token_ids.get()),
                kTopK,
                static_cast<float *>(probabilities.get()),
                static_cast<float *>(scratch_values.get()),
                static_cast<int *>(scratch_indices.get()),
                scratch_capacity,
                static_cast<const int *>(active_rows.get()),
                /*device_idx=*/0,
                stream);
        };

        /*
         * Establish launch validity before capture.  This synchronization is a
         * fixture setup boundary and is deliberately outside the timed graph.
         */
        ASSERT_TRUE(launch());
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        HipSamplingGraph graph;
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

        const int warmup_iterations = samplingPerfWarmupIterations();
        for (int warmup = 0; warmup < warmup_iterations; ++warmup)
            ASSERT_EQ(hipGraphLaunch(graph.executable(), stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        ASSERT_EQ(hipEventCreate(&start), hipSuccess);
        ASSERT_EQ(hipEventCreate(&stop), hipSuccess);
        ASSERT_EQ(hipEventRecord(start, stream), hipSuccess);
        const int iterations = samplingPerfIterations();
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

/**
 * @brief Report production Qwen stochastic-verifier distribution economy.
 */
TEST(Perf__ROCmSampling, Qwen36TopK40VerifierRowsGraphReplay)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        GTEST_SKIP() << "No ROCm device available";

    constexpr std::array<int, 5> kSupportedRowCounts = {1, 2, 4, 8, 16};
    std::vector<int> row_counts(
        kSupportedRowCounts.begin(),
        kSupportedRowCounts.end());
    if (const char *raw =
            std::getenv("LLAMINAR_ROCM_SAMPLING_PERF_ROWS");
        raw && *raw)
    {
        const int requested = std::atoi(raw);
        ASSERT_NE(
            std::find(
                kSupportedRowCounts.begin(),
                kSupportedRowCounts.end(),
                requested),
            kSupportedRowCounts.end())
            << "LLAMINAR_ROCM_SAMPLING_PERF_ROWS must be one of "
               "1,2,4,8,16";
        row_counts.assign(1, requested);
    }

    std::vector<double> average_us(row_counts.size(), 0.0);
    for (size_t index = 0; index < row_counts.size(); ++index)
    {
        benchmarkTopKTopPVerifierRows(
            row_counts[index],
            &average_us[index]);
        ASSERT_GT(average_us[index], 0.0);
    }

    std::cout
        << "backend,rows,vocab_size,top_k,graph_replay_us,"
           "vocabulary_values_per_us\n";
    for (size_t index = 0; index < row_counts.size(); ++index)
    {
        const double values =
            static_cast<double>(row_counts[index]) * kVocabSize;
        std::cout << std::fixed << std::setprecision(3)
                  << "rocm," << row_counts[index] << ','
                  << kVocabSize << ',' << kTopK << ','
                  << average_us[index] << ','
                  << values / average_us[index] << '\n';
    }
#endif
}
