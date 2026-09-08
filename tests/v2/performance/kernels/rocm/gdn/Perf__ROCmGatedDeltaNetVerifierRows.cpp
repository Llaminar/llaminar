/**
 * @file Perf__ROCmGatedDeltaNetVerifierRows.cpp
 * @brief Release microbenchmarks for ROCm GDN verifier recurrence and state publication.
 *
 * This suite invokes the same exported C ABI used while constructing the
 * production ROCm graph.  All allocations, initialization, and event creation
 * happen before timing begins.  The measured regions therefore contain only
 * explicit-stream kernel launches, while the terminal event synchronization is
 * confined to result collection.
 *
 * M=1 measures ordinary decode recurrence.  M={2,4,8,16} compares one grouped
 * verifier launch against equivalent scalar launches.  The publication case
 * measures the exact device-selected 2 MiB Qwen3.6 recurrent-state copy used
 * after stochastic verification.  Byte equivalence and captured replay are
 * owned by the corresponding ROCm integration sweep; this file owns economy
 * and provides a clean single-kernel attachment point for rocprof.
 * Device-counted cases hold the live prefix fixed while varying the retained
 * capture capacity, separating useful recurrence from inactive-row overhead.
 */

#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    /** @brief Throw a precise diagnostic for a failed HIP runtime operation. */
    void checkHip(hipError_t status, const char *operation)
    {
        if (status != hipSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " + hipGetErrorString(status));
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

    /** @brief Select the benchmark device before any HIP object is created. */
    class ROCmDeviceSelection
    {
    public:
        ROCmDeviceSelection()
        {
            checkHip(hipSetDevice(0), "hipSetDevice");
        }
    };

    /** @brief Persistent device allocation of FP32 elements. */
    class DeviceFloatBuffer
    {
    public:
        /** @brief Allocate and initialize on the fixture's exact producer stream. */
        DeviceFloatBuffer(size_t count, hipStream_t producer_stream) : count_(count)
        {
            if (!producer_stream)
                throw std::invalid_argument("DeviceFloatBuffer requires an explicit stream");
            checkHip(
                hipMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(float)),
                "hipMalloc(DeviceFloatBuffer)");
            checkHip(
                hipMemsetAsync(data_, 0, count_ * sizeof(float), producer_stream),
                "hipMemsetAsync(DeviceFloatBuffer)");
        }

        ~DeviceFloatBuffer()
        {
            if (data_)
                (void)hipFree(data_);
        }

        DeviceFloatBuffer(const DeviceFloatBuffer &) = delete;
        DeviceFloatBuffer &operator=(const DeviceFloatBuffer &) = delete;

        [[nodiscard]] float *get() const { return data_; }

    private:
        float *data_ = nullptr;
        size_t count_ = 0;
    };

    /** @brief Persistent device allocation containing one immutable integer. */
    class DeviceIntScalar
    {
    public:
        /** @brief Publish immutable input before any timed/captured execution. */
        DeviceIntScalar(int value, hipStream_t producer_stream)
        {
            if (!producer_stream)
                throw std::invalid_argument("DeviceIntScalar requires an explicit stream");
            checkHip(
                hipMalloc(reinterpret_cast<void **>(&data_), sizeof(int)),
                "hipMalloc(DeviceIntScalar)");
            checkHip(
                hipMemcpyAsync(data_, &value, sizeof(int), hipMemcpyHostToDevice, producer_stream),
                "hipMemcpyAsync(DeviceIntScalar)");
            // Initialization owns this stack value until its transfer completes.
            // No publication or host wait occurs inside the measured replay.
            checkHip(hipStreamSynchronize(producer_stream), "initialize immutable scalar");
        }

        ~DeviceIntScalar()
        {
            if (data_)
                (void)hipFree(data_);
        }

        DeviceIntScalar(const DeviceIntScalar &) = delete;
        DeviceIntScalar &operator=(const DeviceIntScalar &) = delete;

        [[nodiscard]] int *get() const { return data_; }

    private:
        int *data_ = nullptr;
    };

    /** @brief Non-default stream and persistent events used by every sample. */
    class ROCmTimingContext
    {
    public:
        ROCmTimingContext()
        {
            checkHip(
                hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags");
            checkHip(hipEventCreate(&start), "hipEventCreate(start)");
            checkHip(hipEventCreate(&stop), "hipEventCreate(stop)");
        }

        ~ROCmTimingContext()
        {
            if (stop)
                (void)hipEventDestroy(stop);
            if (start)
                (void)hipEventDestroy(start);
            if (stream)
                (void)hipStreamDestroy(stream);
        }

        ROCmTimingContext(const ROCmTimingContext &) = delete;
        ROCmTimingContext &operator=(const ROCmTimingContext &) = delete;

        hipStream_t stream = nullptr;
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
    };

    /** @brief Retain one exact launch graph until all timed replays finish. */
    class RetainedHipGraph
    {
    public:
        /** @brief Begin with no runtime resources; record() owns materialization. */
        RetainedHipGraph() = default;
        /** @brief Release only after the timing fixture has joined its terminal event. */
        ~RetainedHipGraph()
        {
            if (executable_) (void)hipGraphExecDestroy(executable_);
            if (graph_) (void)hipGraphDestroy(graph_);
        }
        RetainedHipGraph(const RetainedHipGraph &) = delete;
        RetainedHipGraph &operator=(const RetainedHipGraph &) = delete;

        /** @brief Record exactly one production operation, ending capture on errors too. */
        template <typename Launch>
        void record(hipStream_t stream, Launch &&launch)
        {
            if (!stream || graph_ || executable_)
                throw std::invalid_argument("RetainedHipGraph requires a fresh graph and explicit stream");
            checkHip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin GDN capture");
            bool launched = false;
            try { launched = launch(); }
            catch (...)
            {
                (void)hipStreamEndCapture(stream, &graph_);
                throw;
            }
            checkHip(hipStreamEndCapture(stream, &graph_), "end GDN capture");
            if (!launched) throw std::runtime_error("captured GDN operation failed");
            checkHip(hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0), "instantiate GDN capture");
        }

        /** @brief Submit immutable captured work to its explicit measurement stream. */
        void replay(hipStream_t stream) const
        {
            if (!stream || !executable_)
                throw std::invalid_argument("RetainedHipGraph replay requires an executable and stream");
            checkHip(hipGraphLaunch(executable_, stream), "replay GDN capture");
        }
    private:
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t executable_ = nullptr;
    };

    /**
     * @brief Persistent Qwen3.6-sized storage for recurrence and publication.
     *
     * The recurrent state contains 32 heads of 128x128 FP32 matrices, exactly
     * 2 MiB per verifier row.  Four capture rows cover the production depth-3
     * transaction: the base row plus three draft rows.  The grouped benchmark
     * retains sixteen rows so it also proves economy through the supported
     * DFlash-style verifier range.
     */
    template <int HeadCount>
    class GdnBenchmarkStorage
    {
    public:
        static constexpr int kHeads = HeadCount;
        static constexpr int kKeyWidth = 128;
        static constexpr int kValueWidth = 128;
        static constexpr int kMaxRows = 16;
        static constexpr int kPublicationRows = 4;
        static constexpr int kStateFloats =
            kHeads * kKeyWidth * kValueWidth;
        static constexpr int kQkRowFloats = kHeads * kKeyWidth;
        static constexpr int kValueRowFloats = kHeads * kValueWidth;

        GdnBenchmarkStorage()
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
              snapshots(static_cast<size_t>(kMaxRows) * kStateFloats, timing.stream),
              accepted_row(kPublicationRows - 1, timing.stream)
        {}

        /** @brief Launch one scalar recurrence row on the benchmark stream. */
        void launchScalar(int row, float *state)
        {
            const bool launched = rocmGDN_recurrent_step(
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
                throw std::runtime_error("rocmGDN_recurrent_step launch failed");
        }

        /** @brief Launch one grouped verifier recurrence with all snapshots. */
        void launchGrouped(int rows)
        {
            const bool launched = rocmGDN_chunk_forward(
                q.get(),
                k.get(),
                v.get(),
                alpha.get(),
                beta.get(),
                a_log.get(),
                dt_bias.get(),
                output.get(),
                grouped_state.get(),
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
                throw std::runtime_error("rocmGDN_chunk_forward launch failed");
        }

        /**
         * @brief Measure one device-counted recurrence at fixed physical capacity.
         * @param physical_rows Captured row/snapshot stride, never inferred from live data.
         * @param live_rows Immutable device-published prefix for this diagnostic point.
         * @param profile_once Select one replay without warmup for isolated attribution.
         * @return Median microseconds from unprofiled complete graph replays, or the
         *         one diagnostic profiler replay when explicitly requested.
         *
         * Initial state and speculative output state are distinct, as in production.
         * Every replay therefore starts with the same state and writes the same
         * snapshots. Count upload, allocation, and graph construction are not timed.
         */
        double timeDeviceCountedUs(int physical_rows, int live_rows, bool profile_once = false)
        {
            if (physical_rows < 1 || physical_rows > kMaxRows ||
                live_rows < 0 || live_rows > physical_rows)
                throw std::invalid_argument("invalid counted GDN capacity/prefix");
            DeviceIntScalar count(live_rows, timing.stream);
            RetainedHipGraph graph;
            graph.record(timing.stream, [&] {
                return rocmGDN_chunk_forward_batched_effective(
                    q.get(), k.get(), v.get(), alpha.get(), beta.get(),
                    a_log.get(), dt_bias.get(), output.get(),
                    scalar_state.get(), grouped_state.get(),
                    physical_rows, 1, physical_rows,
                    kHeads, kKeyWidth, kValueWidth, true, count.get(),
                    snapshots.get(), kStateFloats, physical_rows, 0, timing.stream);
            });
            std::vector<double> samples;
            const int sample_count = profile_once ? 1 : 31;
            samples.reserve(sample_count);
            for (int sample = 0; sample < sample_count; ++sample)
                samples.push_back(timeAverageUs(
                    !profile_once && sample == 0 ? 20 : 0,
                    profile_once ? 1 : 10,
                    [&] { graph.replay(timing.stream); }));
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
        }

        /** @brief Publish the device-selected accepted recurrent-state row. */
        void launchAcceptedStatePublication()
        {
            /*
             * SingleDevice and ExpertParallel use one GDN geometry, so the
             * cache arena aliases scalar live state to request slot zero. The
             * performance case must exercise that production ownership model,
             * not the distinct-bank LocalTP geometry covered by integration
             * correctness tests.
             */
            const bool launched = rocmGDN_gpu_publish_capture_row_from_device_index(
                grouped_state.get(),
                grouped_state.get(),
                snapshots.get(),
                accepted_row.get(),
                kPublicationRows,
                kStateFloats,
                /*device_idx=*/0,
                timing.stream);
            if (!launched)
            {
                throw std::runtime_error(
                    "rocmGDN_gpu_publish_capture_row_from_device_index launch failed");
            }
        }

        /**
         * @brief Measure average GPU time with no allocation or blocking copy.
         *
         * The one host wait occurs after the terminal timing event and is
         * intentionally outside the production-style launch loop.  It is the
         * benchmark result boundary, not an execution dependency.
         */
        template <typename Launch>
        double timeAverageUs(int warmups, int iterations, Launch &&launch)
        {
            for (int i = 0; i < warmups; ++i)
                launch();
            checkHip(
                hipEventRecord(timing.start, timing.stream),
                "hipEventRecord(start)");
            for (int i = 0; i < iterations; ++i)
                launch();
            checkHip(
                hipEventRecord(timing.stop, timing.stream),
                "hipEventRecord(stop)");
            checkHip(
                hipEventSynchronize(timing.stop),
                "hipEventSynchronize(stop)");
            float elapsed_ms = 0.0f;
            checkHip(
                hipEventElapsedTime(&elapsed_ms, timing.start, timing.stop),
                "hipEventElapsedTime");
            return static_cast<double>(elapsed_ms) * 1000.0 /
                   static_cast<double>(iterations);
        }

        ROCmDeviceSelection device;
        ROCmTimingContext timing;
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
        DeviceIntScalar accepted_row;
    };

    using GdnBenchmarkFixture = GdnBenchmarkStorage<32>;

    /**
     * @brief Exact Qwen3.6-35B-A3B prefill recurrence fixture.
     *
     * The production model owns 32 value heads with 128 key and value
     * dimensions and the dashboard prompt contains 425 real rows.  Keeping
     * this case separate from the verifier fixture avoids allocating a
     * 425-row state-snapshot matrix: ordinary prefill commits only the live
     * terminal state and passes no speculative capture rows.
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
              output(static_cast<size_t>(kRows) * kValueRowFloats, timing.stream),
              state(kStateFloats, timing.stream)
        {}

        /** @brief Launch the exact production prefill recurrence once. */
        void launch()
        {
            const bool launched = rocmGDN_chunk_forward(
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
                throw std::runtime_error("rocmGDN_chunk_forward prefill launch failed");
        }

        /** @brief Measure repeated device execution behind one terminal event. */
        double timeAverageUs(int warmups, int iterations)
        {
            for (int i = 0; i < warmups; ++i)
                launch();
            checkHip(
                hipEventRecord(timing.start, timing.stream),
                "hipEventRecord(prefill start)");
            for (int i = 0; i < iterations; ++i)
                launch();
            checkHip(
                hipEventRecord(timing.stop, timing.stream),
                "hipEventRecord(prefill stop)");
            checkHip(
                hipEventSynchronize(timing.stop),
                "hipEventSynchronize(prefill stop)");
            float elapsed_ms = 0.0f;
            checkHip(
                hipEventElapsedTime(&elapsed_ms, timing.start, timing.stop),
                "hipEventElapsedTime(prefill)");
            return static_cast<double>(elapsed_ms) * 1000.0 /
                   static_cast<double>(iterations);
        }

        ROCmDeviceSelection device;
        ROCmTimingContext timing;
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

/** @brief Isolate Qwen3.8-27B's 48-head recurrence capacity tax without model setup. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, Qwen38DeviceCountedCapacity)
{
    GdnBenchmarkStorage<48> fixture;
    std::cout << "backend,case,physical_rows,live_rows,heads,d_k,d_v,median_us\n";
    for (int live : {2, 3})
        for (int capacity : {3, 4, 8, 16, 3})
        {
            const double elapsed = fixture.timeDeviceCountedUs(capacity, live);
            EXPECT_GT(elapsed, 0.0);
            std::cout << "rocm,gdn_counted," << capacity << ',' << live
                      << ",48,128,128," << elapsed << '\n';
        }
}

/** @brief Attach rocprof to exactly one declared captured recurrence shape. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, Qwen38DeviceCountedProfiler)
{
    const int capacity = static_cast<int>(positiveEnv("LLAMINAR_ROCM_GDN_PROFILE_CAPACITY", 16));
    const int live = static_cast<int>(positiveEnv("LLAMINAR_ROCM_GDN_PROFILE_LIVE_ROWS", 3));
    GdnBenchmarkStorage<48> fixture;
    const double elapsed = fixture.timeDeviceCountedUs(capacity, live, true);
    EXPECT_GT(elapsed, 0.0);
    std::cout << "rocm,gdn_counted_profile," << capacity << ',' << live
              << ",48,128,128," << elapsed << '\n';
}

/** @brief Gate the exact 425-row Qwen3.6 production prefill recurrence. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, Qwen36ProductionPrefillM425)
{
    GdnPrefillBenchmarkFixture fixture;
    const double average_us = fixture.timeAverageUs(
        /*warmups=*/10,
        /*iterations=*/50);

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,heads,d_k,d_v,avg_us\n"
              << "rocm,gdn_prefill,"
              << GdnPrefillBenchmarkFixture::kRows << ','
              << GdnPrefillBenchmarkFixture::kHeads << ','
              << GdnPrefillBenchmarkFixture::kKeyWidth << ','
              << GdnPrefillBenchmarkFixture::kValueWidth << ','
              << average_us << '\n';
    EXPECT_LT(
        average_us,
        positiveEnv("LLAMINAR_ROCM_GDN_PREFILL_M425_MAX_US", 5000.0));
}

/**
 * @brief Expose exactly one production Qwen 3.6 M=425 prefill dispatch to rocprof.
 *
 * Counter collection must describe one kernel candidate, not a timing loop or
 * a mixture of warmup and measurement launches. The fixture binds persistent
 * buffers before this test enters its one-iteration timing boundary, allowing
 * rocprof to attach occupancy, VALU, LDS, and scratch evidence to precisely the
 * specialized long-prefill kernel certified by the byte-equivalence suite.
 */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, ProfilerAttachmentQwen36PrefillM425)
{
    GdnPrefillBenchmarkFixture fixture;
    const double elapsed_us = fixture.timeAverageUs(
        /*warmups=*/0,
        /*iterations=*/1);

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,heads,d_k,d_v,profiled_us\n"
              << "rocm,gdn_prefill_profiler_attachment,"
              << GdnPrefillBenchmarkFixture::kRows << ','
              << GdnPrefillBenchmarkFixture::kHeads << ','
              << GdnPrefillBenchmarkFixture::kKeyWidth << ','
              << GdnPrefillBenchmarkFixture::kValueWidth << ','
              << elapsed_us << '\n';
    // Profiling intentionally perturbs event duration; the repeated case above
    // owns latency, while this assertion proves the isolated dispatch completed.
    EXPECT_GT(elapsed_us, 0.0);
}

/** @brief Gate the production M=1 ROCm recurrent-step latency. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, M1Decode)
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
              << "rocm,gdn_recurrent,1," << average_us << '\n';
    EXPECT_LT(
        average_us,
        positiveEnv("LLAMINAR_ROCM_GDN_M1_MAX_US", 25.0));
}

/** @brief Require grouped ROCm verifier recurrence to beat scalar replay. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, GroupedM2M4M8M16)
{
    GdnBenchmarkFixture fixture;
    constexpr std::array<int, 4> kRows = {2, 4, 8, 16};
    constexpr std::array<double, 4> kDefaultMinimumSpeedups = {
        1.02, 1.35, 1.50, 1.65};
    constexpr std::array<const char *, 4> kSpeedupEnvironment = {
        "LLAMINAR_ROCM_GDN_GROUPED_M2_MIN_SPEEDUP",
        "LLAMINAR_ROCM_GDN_GROUPED_M4_MIN_SPEEDUP",
        "LLAMINAR_ROCM_GDN_GROUPED_M8_MIN_SPEEDUP",
        "LLAMINAR_ROCM_GDN_GROUPED_M16_MIN_SPEEDUP"};
    constexpr std::array<double, 4> kDefaultMaximumLatencyUs = {
        35.0, 50.0, 90.0, 165.0};
    constexpr std::array<const char *, 4> kLatencyEnvironment = {
        "LLAMINAR_ROCM_GDN_GROUPED_M2_MAX_US",
        "LLAMINAR_ROCM_GDN_GROUPED_M4_MAX_US",
        "LLAMINAR_ROCM_GDN_GROUPED_M8_MAX_US",
        "LLAMINAR_ROCM_GDN_GROUPED_M16_MAX_US"};
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
                  << "rocm,gdn_grouped," << rows << ','
                  << grouped_us << ',' << serial_us << ',' << speedup << '\n';
        EXPECT_GT(
            speedup,
            positiveEnv(
                kSpeedupEnvironment[index],
                kDefaultMinimumSpeedups[index]))
            << "M=" << rows;
        EXPECT_LT(
            grouped_us,
            positiveEnv(
                kLatencyEnvironment[index],
                kDefaultMaximumLatencyUs[index]))
            << "M=" << rows;
    }
}

/** @brief Gate device-selected publication of one 2 MiB recurrent-state row. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, AcceptedStatePublication)
{
    GdnBenchmarkFixture fixture;
    constexpr int kWarmups = 100;
    constexpr int kIterations = 1000;
    const double average_us = fixture.timeAverageUs(
        kWarmups,
        kIterations,
        [&]() { fixture.launchAcceptedStatePublication(); });

    const double moved_bytes =
        2.0 * static_cast<double>(GdnBenchmarkFixture::kStateFloats) *
        static_cast<double>(sizeof(float));
    const double effective_gib_per_second =
        moved_bytes / (average_us * 1.0e-6) /
        static_cast<double>(1ull << 30);

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,state_bytes,avg_us,effective_gib_per_s\n"
              << "rocm,gdn_accepted_state_publication,"
              << GdnBenchmarkFixture::kStateFloats * sizeof(float) << ','
              << average_us << ',' << effective_gib_per_second << '\n';
    EXPECT_LT(
        average_us,
        positiveEnv("LLAMINAR_ROCM_GDN_PUBLICATION_MAX_US", 10.0));
}

/**
 * @brief Expose exactly one production M=4 recurrence to rocprof.
 *
 * This test deliberately performs no warmup launch. The fixture initializes
 * persistent storage first, then the timing region contains one and only one
 * grouped recurrence dispatch. Counter collection can therefore attach to this
 * filter without mixing M values or scalar-reference launches.
 */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, ProfilerAttachmentM4)
{
    GdnBenchmarkFixture fixture;
    const double average_us = fixture.timeAverageUs(
        /*warmups=*/0,
        /*iterations=*/1,
        [&]() { fixture.launchGrouped(4); });

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,M,avg_us\n"
              << "rocm,gdn_grouped_profiler_attachment,4,"
              << average_us << '\n';
    // Profiler interception deliberately perturbs HIP event duration. Economy
    // is gated by GroupedM2M4M8M16; this case certifies one successful launch.
    EXPECT_GT(average_us, 0.0);
}

/** @brief Expose exactly one accepted-state publication dispatch to rocprof. */
TEST(Perf__ROCmGatedDeltaNetVerifierRows, ProfilerAttachmentPublication)
{
    GdnBenchmarkFixture fixture;
    const double average_us = fixture.timeAverageUs(
        /*warmups=*/0,
        /*iterations=*/1,
        [&]() { fixture.launchAcceptedStatePublication(); });

    std::cout << std::fixed << std::setprecision(3)
              << "backend,case,state_bytes,avg_us\n"
              << "rocm,gdn_publication_profiler_attachment,"
              << GdnBenchmarkFixture::kStateFloats * sizeof(float) << ','
              << average_us << '\n';
    // AcceptedStatePublication owns the unprofiled latency gate.
    EXPECT_GT(average_us, 0.0);
}
