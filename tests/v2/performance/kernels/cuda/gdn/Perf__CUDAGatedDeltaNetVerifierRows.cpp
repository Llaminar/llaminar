/**
 * @file Perf__CUDAGatedDeltaNetVerifierRows.cpp
 * @brief Release microbenchmarks for CUDA GDN decode, grouped verification, and prefill.
 *
 * The benchmark invokes the same exported entrypoints used by production graph
 * construction. Device buffers, stream, and timing events are persistent; the
 * timed region contains only recurrence launches. M=1 measures ordinary decode,
 * The captured M=1 lane times retained GPU work without per-launch host tax.
 * M={2,4,8,16} compares one grouped verifier launch with the equivalent
 * sequence of scalar launches. The M=425 fixture isolates the exact Qwen 3.6
 * long-prefill geometry for latency and Nsight attachment. Correctness and
 * graph replay are certified by the CUDA GDN integration suite; this file owns
 * the complementary economy gate.
 */

#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "kernels/cuda/gdn/CUDAGatedDeltaNet.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    /** @brief Throw a precise diagnostic for a failed CUDA runtime operation. */
    void checkCuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }

    /** @brief Read a positive floating-point performance override. */
    double positiveEnv(const char *name, double fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        const double parsed = std::strtod(raw, nullptr);
        return parsed > 0.0 ? parsed : fallback;
    }

    /** @brief Select the benchmark device before any stream or buffer exists. */
    class CudaDeviceSelection
    {
    public:
        CudaDeviceSelection()
        {
            checkCuda(cudaSetDevice(0), "cudaSetDevice");
        }
    };

    /** @brief Device allocation whose lifetime strictly encloses all launches. */
    class DeviceFloatBuffer
    {
    public:
        DeviceFloatBuffer(size_t count, cudaStream_t producer_stream)
            : count_(count)
        {
            if (!producer_stream)
                throw std::invalid_argument(
                    "DeviceFloatBuffer requires an explicit producer stream");
            checkCuda(
                cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(float)),
                "cudaMalloc(DeviceFloatBuffer)");
            checkCuda(
                cudaMemsetAsync(
                    data_, 0, count_ * sizeof(float), producer_stream),
                "cudaMemsetAsync(DeviceFloatBuffer)");
        }

        ~DeviceFloatBuffer()
        {
            if (data_)
                (void)cudaFree(data_);
        }

        DeviceFloatBuffer(const DeviceFloatBuffer &) = delete;
        DeviceFloatBuffer &operator=(const DeviceFloatBuffer &) = delete;

        [[nodiscard]] float *get() const { return data_; }

    private:
        float *data_ = nullptr;
        size_t count_ = 0;
    };

    /** @brief Non-default stream and event pair used by every timed sample. */
    class CudaTimingContext
    {
    public:
        CudaTimingContext()
        {
            checkCuda(
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
            checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
            checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        }

        ~CudaTimingContext()
        {
            if (stop)
                (void)cudaEventDestroy(stop);
            if (start)
                (void)cudaEventDestroy(start);
            if (stream)
                (void)cudaStreamDestroy(stream);
        }

        CudaTimingContext(const CudaTimingContext &) = delete;
        CudaTimingContext &operator=(const CudaTimingContext &) = delete;

        cudaStream_t stream = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
    };

    /**
     * @brief Persistent Qwen3.6-sized buffers shared by scalar and grouped runs.
     */
    class GdnBenchmarkFixture
    {
    public:
        static constexpr int kHeads = 32;
        static constexpr int kKeyWidth = 128;
        static constexpr int kValueWidth = 128;
        static constexpr int kMaxRows = 16;
        static constexpr int kStateFloats =
            kHeads * kKeyWidth * kValueWidth;
        static constexpr int kQkRowFloats = kHeads * kKeyWidth;
        static constexpr int kValueRowFloats = kHeads * kValueWidth;

        GdnBenchmarkFixture()
            : q(static_cast<size_t>(kMaxRows) * kQkRowFloats, timing.stream),
              k(static_cast<size_t>(kMaxRows) * kQkRowFloats, timing.stream),
              v(static_cast<size_t>(kMaxRows) * kValueRowFloats, timing.stream),
              alpha(static_cast<size_t>(kMaxRows) * kHeads, timing.stream),
              beta(static_cast<size_t>(kMaxRows) * kHeads, timing.stream),
              a_log(kHeads, timing.stream),
              dt_bias(kHeads, timing.stream),
              output(static_cast<size_t>(kMaxRows) * kValueRowFloats, timing.stream),
              scalar_state(kStateFloats, timing.stream),
              grouped_state(kStateFloats, timing.stream),
              snapshots(static_cast<size_t>(kMaxRows) * kStateFloats, timing.stream)
        {}

        /** @brief Launch one scalar row on the benchmark stream. */
        void launchScalar(int row, float *state)
        {
            const bool launched = cudaGDN_recurrent_step(
                q.get() + static_cast<size_t>(row) * kQkRowFloats,
                k.get() + static_cast<size_t>(row) * kQkRowFloats,
                v.get() + static_cast<size_t>(row) * kValueRowFloats,
                alpha.get() + static_cast<size_t>(row) * kHeads,
                beta.get() + static_cast<size_t>(row) * kHeads,
                a_log.get(),
                dt_bias.get(),
                output.get() + static_cast<size_t>(row) * kValueRowFloats,
                state,
                state,
                kHeads,
                kKeyWidth,
                kValueWidth,
                /*use_qk_l2norm=*/true,
                /*device_idx=*/0,
                timing.stream);
            if (!launched)
                throw std::runtime_error("cudaGDN_recurrent_step launch failed");
        }

        /** @brief Launch one grouped verifier transaction with full snapshots. */
        void launchGrouped(int rows)
        {
            const bool launched = cudaGDN_chunk_forward(
                q.get(),
                k.get(),
                v.get(),
                alpha.get(),
                beta.get(),
                a_log.get(),
                dt_bias.get(),
                output.get(),
                scalar_state.get(),
                grouped_state.get(),
                rows,
                kHeads,
                kKeyWidth,
                kValueWidth,
                /*use_qk_l2norm=*/true,
                snapshots.get(),
                kStateFloats,
                rows,
                /*device_idx=*/0,
                timing.stream);
            if (!launched)
                throw std::runtime_error("cudaGDN_chunk_forward launch failed");
        }

        /** @brief Time a callable without allocations or whole-device syncs. */
        template <typename Launch>
        double timeAverageUs(int warmups, int iterations, Launch &&launch)
        {
            for (int i = 0; i < warmups; ++i)
                launch();
            checkCuda(
                cudaEventRecord(timing.start, timing.stream),
                "cudaEventRecord(start)");
            for (int i = 0; i < iterations; ++i)
                launch();
            checkCuda(
                cudaEventRecord(timing.stop, timing.stream),
                "cudaEventRecord(stop)");
            checkCuda(cudaEventSynchronize(timing.stop), "cudaEventSynchronize(stop)");
            float elapsed_ms = 0.0f;
            checkCuda(
                cudaEventElapsedTime(&elapsed_ms, timing.start, timing.stop),
                "cudaEventElapsedTime");
            return static_cast<double>(elapsed_ms) * 1000.0 /
                   static_cast<double>(iterations);
        }

        /**
         * @brief Measure a retained M1 launch chain, independently of host enqueue.
         * @return Median unprofiled per-step latency in microseconds.
         *
         * The fixture's zero state is a recurrence fixed point, so repeated
         * launches execute the same operations and addresses. Integration
         * tests separately authenticate nonzero state and every output byte.
         * Capture, allocation, warmup and result synchronization are untimed.
         */
        double timeCapturedScalarUs()
        {
            /** @brief Release this test's captured topology before its buffers. */
            struct GraphOwner
            {
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t executable = nullptr;
                /** @brief Retire the executable before its graph definition. */
                ~GraphOwner()
                {
                    if (executable) (void)cudaGraphExecDestroy(executable);
                    if (graph) (void)cudaGraphDestroy(graph);
                }
            } graph;
            constexpr int steps = 128;
            checkCuda(cudaStreamBeginCapture(timing.stream, cudaStreamCaptureModeThreadLocal),
                      "cudaStreamBeginCapture(GDN economy)");
            for (int step = 0; step < steps; ++step)
                launchScalar(0, scalar_state.get());
            checkCuda(cudaStreamEndCapture(timing.stream, &graph.graph),
                      "cudaStreamEndCapture(GDN economy)");
            checkCuda(cudaGraphInstantiate(&graph.executable, graph.graph, 0),
                      "cudaGraphInstantiate(GDN economy)");
            std::array<double, 20> samples{};
            for (int round = -5; round < static_cast<int>(samples.size()); ++round)
            {
                checkCuda(cudaEventRecord(timing.start, timing.stream), "record graph start");
                checkCuda(cudaGraphLaunch(graph.executable, timing.stream), "launch GDN graph");
                checkCuda(cudaEventRecord(timing.stop, timing.stream), "record graph stop");
                checkCuda(cudaEventSynchronize(timing.stop), "observe GDN graph timing");
                float ms = 0.0f;
                checkCuda(cudaEventElapsedTime(&ms, timing.start, timing.stop), "read GDN graph timing");
                if (round >= 0) samples[round] = ms * 1000.0 / steps;
            }
            std::sort(samples.begin(), samples.end());
            return (samples[9] + samples[10]) / 2.0;
        }

        CudaDeviceSelection device;
        CudaTimingContext timing;
        DeviceFloatBuffer q;
        DeviceFloatBuffer k;
        DeviceFloatBuffer v;
        DeviceFloatBuffer alpha;
        DeviceFloatBuffer beta;
        DeviceFloatBuffer a_log;
        DeviceFloatBuffer dt_bias;
        DeviceFloatBuffer output;
        DeviceFloatBuffer scalar_state;
        DeviceFloatBuffer grouped_state;
        DeviceFloatBuffer snapshots;
    };

    /**
     * @brief Exact Qwen3.6-35B-A3B long-prefill recurrence fixture.
     *
     * Every allocation and initialization precedes the measured region. The
     * production entrypoint launches its mandatory preprocessing node followed
     * by the width-specialized recurrent node on one explicit stream. No state
     * snapshots are requested because ordinary prefill publishes only the
     * terminal live state.
     */
    class GdnPrefillBenchmarkFixture
    {
    public:
        static constexpr int kRows = 425;
        static constexpr int kHeads = 32;
        static constexpr int kKeyWidth = 128;
        static constexpr int kValueWidth = 128;
        static constexpr int kStateFloats =
            kHeads * kKeyWidth * kValueWidth;
        static constexpr int kQkRowFloats = kHeads * kKeyWidth;
        static constexpr int kValueRowFloats = kHeads * kValueWidth;

        GdnPrefillBenchmarkFixture()
            : q(static_cast<size_t>(kRows) * kQkRowFloats, timing.stream),
              k(static_cast<size_t>(kRows) * kQkRowFloats, timing.stream),
              v(static_cast<size_t>(kRows) * kValueRowFloats, timing.stream),
              alpha(static_cast<size_t>(kRows) * kHeads, timing.stream),
              beta(static_cast<size_t>(kRows) * kHeads, timing.stream),
              a_log(kHeads, timing.stream),
              dt_bias(kHeads, timing.stream),
              output(
                  static_cast<size_t>(kRows) * kValueRowFloats,
                  timing.stream),
              state(kStateFloats, timing.stream)
        {}

        /** @brief Launch one exact production-shape long-prefill recurrence. */
        void launch()
        {
            const bool launched = cudaGDN_chunk_forward(
                q.get(),
                k.get(),
                v.get(),
                alpha.get(),
                beta.get(),
                a_log.get(),
                dt_bias.get(),
                output.get(),
                state.get(),
                state.get(),
                kRows,
                kHeads,
                kKeyWidth,
                kValueWidth,
                /*use_qk_l2norm=*/true,
                /*state_snapshots=*/nullptr,
                /*snapshot_stride_floats=*/0,
                /*max_snapshot_rows=*/0,
                /*device_idx=*/0,
                timing.stream);
            if (!launched)
            {
                throw std::runtime_error(
                    "cudaGDN_chunk_forward prefill launch failed");
            }
        }

        /** @brief Measure repeated launches behind one terminal timing event. */
        double timeAverageUs(int warmups, int iterations)
        {
            for (int i = 0; i < warmups; ++i)
                launch();
            checkCuda(
                cudaEventRecord(timing.start, timing.stream),
                "cudaEventRecord(prefill start)");
            for (int i = 0; i < iterations; ++i)
                launch();
            checkCuda(
                cudaEventRecord(timing.stop, timing.stream),
                "cudaEventRecord(prefill stop)");
            checkCuda(
                cudaEventSynchronize(timing.stop),
                "cudaEventSynchronize(prefill stop)");
            float elapsed_ms = 0.0f;
            checkCuda(
                cudaEventElapsedTime(&elapsed_ms, timing.start, timing.stop),
                "cudaEventElapsedTime(prefill)");
            return static_cast<double>(elapsed_ms) * 1000.0 /
                   static_cast<double>(iterations);
        }

        CudaDeviceSelection device;
        CudaTimingContext timing;
        DeviceFloatBuffer q;
        DeviceFloatBuffer k;
        DeviceFloatBuffer v;
        DeviceFloatBuffer alpha;
        DeviceFloatBuffer beta;
        DeviceFloatBuffer a_log;
        DeviceFloatBuffer dt_bias;
        DeviceFloatBuffer output;
        DeviceFloatBuffer state;
    };
} // namespace

/** @brief Gate the exact Qwen 3.6 M=425 CUDA long-prefill recurrence. */
TEST(Perf__CUDAGatedDeltaNetVerifierRows, Qwen36ProductionPrefillM425)
{
    GdnPrefillBenchmarkFixture fixture;
    const double average_us = fixture.timeAverageUs(
        /*warmups=*/10,
        /*iterations=*/50);

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,heads,d_k,d_v,avg_us\n"
              << "cuda,gdn_prefill,"
              << GdnPrefillBenchmarkFixture::kRows << ','
              << GdnPrefillBenchmarkFixture::kHeads << ','
              << GdnPrefillBenchmarkFixture::kKeyWidth << ','
              << GdnPrefillBenchmarkFixture::kValueWidth << ','
              << average_us << '\n';
    EXPECT_LT(
        average_us,
        positiveEnv("LLAMINAR_CUDA_GDN_PREFILL_M425_MAX_US", 5000.0));
}

/**
 * @brief Expose one exact Qwen 3.6 M=425 CUDA prefill launch to Nsight.
 *
 * Nsight Compute counter collection must attach to one candidate invocation,
 * not a warmup or timing loop. The production wrapper emits exactly one
 * preprocessing launch and one width-specialized recurrence launch; a kernel
 * name filter can therefore isolate either node without unrelated GPU work.
 */
TEST(Perf__CUDAGatedDeltaNetVerifierRows, ProfilerAttachmentQwen36PrefillM425)
{
    GdnPrefillBenchmarkFixture fixture;
    const double elapsed_us = fixture.timeAverageUs(
        /*warmups=*/0,
        /*iterations=*/1);

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,heads,d_k,d_v,profiled_us\n"
              << "cuda,gdn_prefill_profiler_attachment,"
              << GdnPrefillBenchmarkFixture::kRows << ','
              << GdnPrefillBenchmarkFixture::kHeads << ','
              << GdnPrefillBenchmarkFixture::kKeyWidth << ','
              << GdnPrefillBenchmarkFixture::kValueWidth << ','
              << elapsed_us << '\n';
    EXPECT_GT(elapsed_us, 0.0);
}

/** @brief Gate the production M=1 CUDA recurrence latency. */
TEST(Perf__CUDAGatedDeltaNetVerifierRows, M1Decode)
{
    GdnBenchmarkFixture fixture;
    constexpr int kWarmups = 50;
    constexpr int kIterations = 500;
    const double average_us = fixture.timeAverageUs(
        kWarmups,
        kIterations,
        [&]() { fixture.launchScalar(0, fixture.scalar_state.get()); });

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,avg_us\n"
              << "cuda,gdn_recurrent,1," << average_us << '\n';
    EXPECT_LT(
        average_us,
        positiveEnv("LLAMINAR_CUDA_GDN_M1_MAX_US", 25.0));
}

/** @brief Expose captured M1 economics without conflating host submission cost. */
TEST(Perf__CUDAGatedDeltaNetVerifierRows, M1CapturedDecode)
{
    GdnBenchmarkFixture fixture;
    const double median_us = fixture.timeCapturedScalarUs();
    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,heads,d_k,d_v,median_us\n"
              << "cuda,gdn_captured,1," << GdnBenchmarkFixture::kHeads << ','
              << GdnBenchmarkFixture::kKeyWidth << ','
              << GdnBenchmarkFixture::kValueWidth << ',' << median_us << '\n';
    EXPECT_GT(median_us, 0.0);
}

/**
 * @brief Require grouped verifier recurrence to beat scalar launch replay.
 */
TEST(Perf__CUDAGatedDeltaNetVerifierRows, GroupedM2M4M8M16)
{
    GdnBenchmarkFixture fixture;
    constexpr std::array<int, 4> kRows = {2, 4, 8, 16};
    // M=2 must publish two complete state snapshots, while scalar decode does
    // not. Near-parity is therefore the correct tiny-M floor; larger M must
    // amortize publication and demonstrate progressively stronger economy.
    constexpr std::array<double, 4> kDefaultMinimumSpeedups = {
        0.90, 1.10, 1.25, 1.40};
    constexpr std::array<const char *, 4> kSpeedupEnvironment = {
        "LLAMINAR_CUDA_GDN_GROUPED_M2_MIN_SPEEDUP",
        "LLAMINAR_CUDA_GDN_GROUPED_M4_MIN_SPEEDUP",
        "LLAMINAR_CUDA_GDN_GROUPED_M8_MIN_SPEEDUP",
        "LLAMINAR_CUDA_GDN_GROUPED_M16_MIN_SPEEDUP"};
    constexpr int kWarmups = 20;
    constexpr int kIterations = 200;

    std::cout << "backend,case,M,grouped_us,serial_us,speedup\n";
    for (size_t index = 0; index < kRows.size(); ++index)
    {
        const int rows = kRows[index];
        const double grouped_us = fixture.timeAverageUs(
            kWarmups,
            kIterations,
            [&]() { fixture.launchGrouped(rows); });
        const double serial_us = fixture.timeAverageUs(
            kWarmups,
            kIterations,
            [&]()
            {
                for (int row = 0; row < rows; ++row)
                    fixture.launchScalar(row, fixture.scalar_state.get());
            });
        const double speedup = serial_us / grouped_us;

        std::cout << std::fixed << std::setprecision(3)
                  << "cuda,gdn_grouped," << rows << ','
                  << grouped_us << ',' << serial_us << ',' << speedup << '\n';
        EXPECT_GT(
            speedup,
            positiveEnv(
                kSpeedupEnvironment[index],
                kDefaultMinimumSpeedups[index]))
            << "M=" << rows;
    }
}
