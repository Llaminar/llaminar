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
 */

#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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
        explicit DeviceFloatBuffer(size_t count) : count_(count)
        {
            checkHip(
                hipMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(float)),
                "hipMalloc(DeviceFloatBuffer)");
            checkHip(
                hipMemset(data_, 0, count_ * sizeof(float)),
                "hipMemset(DeviceFloatBuffer)");
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
        explicit DeviceIntScalar(int value)
        {
            checkHip(
                hipMalloc(reinterpret_cast<void **>(&data_), sizeof(int)),
                "hipMalloc(DeviceIntScalar)");
            checkHip(
                hipMemcpy(data_, &value, sizeof(int), hipMemcpyHostToDevice),
                "hipMemcpy(DeviceIntScalar)");
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

    /**
     * @brief Persistent Qwen3.6-sized storage for recurrence and publication.
     *
     * The recurrent state contains 32 heads of 128x128 FP32 matrices, exactly
     * 2 MiB per verifier row.  Four capture rows cover the production depth-3
     * transaction: the base row plus three draft rows.  The grouped benchmark
     * retains sixteen rows so it also proves economy through the supported
     * DFlash-style verifier range.
     */
    class GdnBenchmarkFixture
    {
    public:
        static constexpr int kHeads = 32;
        static constexpr int kKeyWidth = 128;
        static constexpr int kValueWidth = 128;
        static constexpr int kMaxRows = 16;
        static constexpr int kPublicationRows = 4;
        static constexpr int kStateFloats =
            kHeads * kKeyWidth * kValueWidth;
        static constexpr int kQkRowFloats = kHeads * kKeyWidth;
        static constexpr int kValueRowFloats = kHeads * kValueWidth;

        GdnBenchmarkFixture()
            : q(static_cast<size_t>(kMaxRows) * kQkRowFloats),
              k(static_cast<size_t>(kMaxRows) * kQkRowFloats),
              v(static_cast<size_t>(kMaxRows) * kValueRowFloats),
              alpha(static_cast<size_t>(kMaxRows) * kHeads),
              beta(static_cast<size_t>(kMaxRows) * kHeads),
              a_log(kHeads),
              dt_bias(kHeads),
              output(static_cast<size_t>(kMaxRows) * kValueRowFloats),
              scalar_state(kStateFloats),
              grouped_state(kStateFloats),
              snapshots(static_cast<size_t>(kMaxRows) * kStateFloats),
              accepted_row(kPublicationRows - 1)
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
} // namespace

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
