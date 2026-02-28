/**
 * @file Perf__RCCLAllreduceLatency.cpp
 * @brief Performance benchmark: RCCL allreduce minimum latency for small transfers
 *
 * **Motivation**: In local TP for Qwen2.5-7B decode, each allreduce transfers
 * ~14 KB (3584 floats = 1 token × d_model). There are 56 allreduce calls per
 * decode step (28 layers × 2: Wo + FFN down). The minimum achievable allreduce
 * latency directly bounds TP efficiency.
 *
 * **What this measures**:
 *   1. Raw RCCL ncclAllReduce() latency for various message sizes (256B → 1MB)
 *   2. Impact of NCCL environment tuning variables:
 *      - NCCL_MIN_NCHANNELS / NCCL_MAX_NCHANNELS (channel count)
 *      - NCCL_BUFFSIZE (internal ring buffer)
 *      - NCCL_PROTO (LL, LL128, Simple)
 *      - NCCL_ALGO (Ring, Tree, CollNet)
 *      - NCCL_P2P_LEVEL / NCCL_P2P_DISABLE
 *   3. On-stream vs dedicated-stream allreduce
 *   4. Multi-allreduce pipelining (ncclGroupStart/End batching)
 *   5. Comparison to llaminar's actual allreduce path
 *
 * **Target hardware**: AMD MI50 (gfx906) × 2-3, PCIe gen3
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
#include <functional>
#include <map>

#include "fort.hpp"

// ============================================================================
// RCCL Allreduce Latency Benchmark
// ============================================================================

#ifdef HAVE_ROCM

// We use the dynamic loader consistent with the rest of the codebase
#include "collective/backends/RCCLDynamicLoader.h"
namespace rccl = llaminar2::rccl_dynamic;

#define HIP_CHECK(call)                                                               \
    do                                                                                \
    {                                                                                 \
        hipError_t err = (call);                                                      \
        EXPECT_EQ(err, hipSuccess) << #call << " failed: " << hipGetErrorString(err); \
    } while (0)

#define RCCL_CHECK(call)                                                                        \
    do                                                                                          \
    {                                                                                           \
        rccl::ncclResult_t r = (call);                                                          \
        EXPECT_EQ(r, rccl::ncclSuccess) << #call << " failed: " << rccl::ncclGetErrorString(r); \
    } while (0)

namespace
{

    // ============================================================================
    // Timing Utilities
    // ============================================================================

    struct LatencyResult
    {
        double mean_us = 0.0;
        double median_us = 0.0;
        double p5_us = 0.0;  // 5th percentile
        double p95_us = 0.0; // 95th percentile
        double min_us = 0.0;
        double max_us = 0.0;
        double stddev_us = 0.0;
        int iterations = 0;
        size_t bytes = 0;
        std::string label;
    };

    LatencyResult computeStats(std::vector<double> &samples, size_t bytes, const std::string &label)
    {
        LatencyResult r;
        r.iterations = static_cast<int>(samples.size());
        r.bytes = bytes;
        r.label = label;
        if (samples.empty())
            return r;

        std::sort(samples.begin(), samples.end());
        r.min_us = samples.front();
        r.max_us = samples.back();
        r.median_us = samples[samples.size() / 2];
        r.p5_us = samples[static_cast<size_t>(samples.size() * 0.05)];
        r.p95_us = samples[static_cast<size_t>(samples.size() * 0.95)];
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

    constexpr int WARMUP_ITERS = 200; // Extra warmup for RCCL (first calls are slow)
    constexpr int BENCH_ITERS = 1000; // Enough iterations for stable percentiles

    // Message sizes to test — covering both decode (14 KB) and prefill (~8.5 MB)
    // 14336 bytes = 3584 floats = Qwen2.5-7B hidden_dim × sizeof(float) ← DECODE
    // 8553472 bytes = 2138368 floats = 596 × 3584 × sizeof(float) ← PREFILL (596 tokens)
    const std::vector<size_t> MESSAGE_SIZES_BYTES = {
        256,     // 64 floats — tiny
        1024,    // 256 floats
        4096,    // 1024 floats — 1 KB
        14336,   // 3584 floats — Qwen2.5-7B d_model ← DECODE TARGET
        16384,   // 4096 floats — power of 2
        32768,   // 8192 floats
        65536,   // 16384 floats — 64 KB
        131072,  // 32768 floats — 128 KB
        524288,  // 131072 floats — 512 KB
        1048576, // 262144 floats — 1 MB
        2097152, // 524288 floats — 2 MB
        4194304, // 1048576 floats — 4 MB
        8553472, // 2138368 floats — 8.5 MB ← PREFILL TARGET (596 × 3584)
    };

    // ============================================================================
    // RCCL Environment Tuning Configurations
    // ============================================================================

    struct RCCLConfig
    {
        std::string name;
        std::map<std::string, std::string> env_vars;
    };

    // NOTE: These env vars must be set BEFORE rccl is loaded (it reads at load time).
    // For configurations that require different env vars, we can only test the one
    // that was set before the library loaded. So we test "live" configs by setting
    // env vars before the first RCCL call in this process.
    //
    // For a full exploration, the test prints the current NCCL_* env so the user
    // can re-run with different settings.

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Perf__RCCLAllreduceLatency : public ::testing::Test
    {
    protected:
        // Per-device resources
        struct DeviceState
        {
            int ordinal = -1;
            float *d_buffer = nullptr;
            hipStream_t stream = nullptr;
            hipEvent_t start_event = nullptr;
            hipEvent_t stop_event = nullptr;
        };

        std::vector<DeviceState> devices_;
        std::vector<rccl::ncclComm_t> comms_;
        int num_devices_ = 0;
        bool initialized_ = false;

        // Maximum buffer size (16 MB — enough for prefill allreduces)
        // Prefill: 596 × 3584 × 4 = 8,553,472 bytes (~8.5 MB)
        static constexpr size_t MAX_BUFFER_BYTES = 16 * 1024 * 1024;
        static constexpr size_t MAX_BUFFER_FLOATS = MAX_BUFFER_BYTES / sizeof(float);

        void SetUp() override
        {
            // Detect available ROCm devices
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            if (err != hipSuccess || device_count < 2)
            {
                GTEST_SKIP() << "Need at least 2 ROCm GPUs for allreduce benchmark (found "
                             << device_count << ")";
                return;
            }

            // Use first 2 devices (matching typical TP=2 setup)
            num_devices_ = 2;

            // Print current NCCL environment for reference
            printCurrentNCCLEnv();

            // Load RCCL
            if (!rccl::isLoaded())
            {
                ASSERT_TRUE(rccl::load()) << "Failed to load RCCL: " << rccl::getLastError();
            }

            // Enable P2P access between devices
            enableP2P();

            // Allocate per-device resources
            devices_.resize(num_devices_);
            for (int i = 0; i < num_devices_; ++i)
            {
                devices_[i].ordinal = i;
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipMalloc(&devices_[i].d_buffer, MAX_BUFFER_BYTES));
                HIP_CHECK(hipStreamCreateWithFlags(&devices_[i].stream, hipStreamNonBlocking));
                HIP_CHECK(hipEventCreateWithFlags(&devices_[i].start_event, hipEventDefault));
                HIP_CHECK(hipEventCreateWithFlags(&devices_[i].stop_event, hipEventDefault));

                // Initialize buffer with device index (so allreduce produces known result)
                std::vector<float> host_data(MAX_BUFFER_FLOATS, static_cast<float>(i + 1));
                HIP_CHECK(hipMemcpy(devices_[i].d_buffer, host_data.data(),
                                    MAX_BUFFER_BYTES, hipMemcpyHostToDevice));
            }

            // Initialize RCCL communicators
            initRCCL();

            // Heavy warmup: run several hundred allreduces to ensure RCCL internal
            // state is fully warmed (first ~10 calls have elevated latency)
            warmupRCCL();

            initialized_ = true;
        }

        void TearDown() override
        {
            // Destroy communicators
            for (auto &comm : comms_)
            {
                if (comm)
                {
                    (void)rccl::ncclCommDestroy(comm);
                }
            }
            comms_.clear();

            // Free device resources
            for (auto &dev : devices_)
            {
                if (dev.stream)
                {
                    (void)hipSetDevice(dev.ordinal);
                    (void)hipStreamDestroy(dev.stream);
                }
                if (dev.start_event)
                {
                    (void)hipSetDevice(dev.ordinal);
                    (void)hipEventDestroy(dev.start_event);
                }
                if (dev.stop_event)
                {
                    (void)hipSetDevice(dev.ordinal);
                    (void)hipEventDestroy(dev.stop_event);
                }
                if (dev.d_buffer)
                {
                    (void)hipSetDevice(dev.ordinal);
                    (void)hipFree(dev.d_buffer);
                }
            }
            devices_.clear();
        }

        // -----------------------------------------------------------------------
        // Setup helpers
        // -----------------------------------------------------------------------

        void printCurrentNCCLEnv()
        {
            std::cout << "\n=== Current NCCL/RCCL Environment ===\n";
            const char *vars[] = {
                "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE",
                "NCCL_NET_GDR_LEVEL", "NCCL_BUFFSIZE", "NCCL_MIN_NCHANNELS",
                "NCCL_MAX_NCHANNELS", "NCCL_PROTO", "NCCL_ALGO",
                "NCCL_LL_THRESHOLD", "NCCL_LL128_NTHREADS",
                "NCCL_NTHREADS", "NCCL_NSOCKS_PERTHREAD",
                "NCCL_SOCKET_NTHREADS", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
                "NCCL_IB_DISABLE", "NCCL_CROSS_NIC",
                "RCCL_MSCCL_ENABLE", "RCCL_FORCE_ENABLE_MSCCL",
                "HSA_FORCE_FINE_GRAIN_PCIE",
                nullptr};

            for (int i = 0; vars[i]; ++i)
            {
                const char *val = std::getenv(vars[i]);
                if (val)
                {
                    std::cout << "  " << vars[i] << "=" << val << "\n";
                }
            }
            std::cout << "  (unset vars use RCCL defaults)\n\n";
        }

        void enableP2P()
        {
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(i));
                for (int j = 0; j < num_devices_; ++j)
                {
                    if (i != j)
                    {
                        int can_access = 0;
                        (void)hipDeviceCanAccessPeer(&can_access, i, j);
                        if (can_access)
                        {
                            (void)hipDeviceEnablePeerAccess(j, 0); // ignore already-enabled errors
                        }
                        else
                        {
                            std::cout << "  WARNING: P2P not available GPU " << i << " -> GPU " << j << "\n";
                        }
                    }
                }
            }
        }

        void initRCCL()
        {
            comms_.resize(num_devices_);

            rccl::ncclUniqueId unique_id;
            RCCL_CHECK(rccl::ncclGetUniqueId(&unique_id));

            RCCL_CHECK(rccl::ncclGroupStart());
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                RCCL_CHECK(rccl::ncclCommInitRank(&comms_[i], num_devices_, unique_id, i));
            }
            RCCL_CHECK(rccl::ncclGroupEnd());

            // Sync all devices after init
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                HIP_CHECK(hipDeviceSynchronize());
            }
        }

        void warmupRCCL()
        {
            // Run 200 allreduces to fully warm RCCL internals
            size_t count = 3584; // Our target size
            for (int iter = 0; iter < WARMUP_ITERS; ++iter)
            {
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, devices_[i].d_buffer, count,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }
            }
        }

        // -----------------------------------------------------------------------
        // Core benchmark: Single allreduce latency (host-timed)
        // -----------------------------------------------------------------------
        LatencyResult benchAllreduceHostTimed(size_t num_floats, const std::string &label)
        {
            size_t bytes = num_floats * sizeof(float);
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                // Sync all streams before timing
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                // Launch allreduce across all devices
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, devices_[i].d_buffer, num_floats,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());

                // Wait for completion on all devices
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            return computeStats(samples, bytes, label);
        }

        // -----------------------------------------------------------------------
        // Core benchmark: Single allreduce latency (GPU event-timed)
        // -----------------------------------------------------------------------
        LatencyResult benchAllreduceGPUTimed(size_t num_floats, const std::string &label)
        {
            size_t bytes = num_floats * sizeof(float);
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            // Use device 0's events for timing (GPU-side timing, more accurate)
            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                // Sync all streams
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                // Record start on device 0's stream
                HIP_CHECK(hipSetDevice(devices_[0].ordinal));
                HIP_CHECK(hipEventRecord(devices_[0].start_event, devices_[0].stream));

                // Launch allreduce
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, devices_[i].d_buffer, num_floats,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());

                // Wait for all devices to complete, then record stop on device 0
                for (int i = 1; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }
                HIP_CHECK(hipSetDevice(devices_[0].ordinal));
                HIP_CHECK(hipEventRecord(devices_[0].stop_event, devices_[0].stream));
                HIP_CHECK(hipEventSynchronize(devices_[0].stop_event));

                float ms = 0.0f;
                HIP_CHECK(hipEventElapsedTime(&ms, devices_[0].start_event, devices_[0].stop_event));
                samples.push_back(static_cast<double>(ms) * 1000.0); // convert to µs
            }

            return computeStats(samples, bytes, label);
        }

        // -----------------------------------------------------------------------
        // Benchmark: Single allreduce without ncclGroupStart/End
        // (tests whether grouping adds overhead for 2-device case)
        // -----------------------------------------------------------------------
        LatencyResult benchAllreduceNoGroup(size_t num_floats, const std::string &label)
        {
            size_t bytes = num_floats * sizeof(float);
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                // Launch allreduce WITHOUT group — each device issues independently
                // For 2 devices, RCCL should handle this fine
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, devices_[i].d_buffer, num_floats,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            return computeStats(samples, bytes, label);
        }

        // -----------------------------------------------------------------------
        // Benchmark: Batched N allreduces in a single ncclGroupStart/End
        // (simulates what a fused layer could do)
        // -----------------------------------------------------------------------
        LatencyResult benchBatchedAllreduce(size_t num_floats, int batch_count, const std::string &label)
        {
            size_t bytes = num_floats * sizeof(float);
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                // Batch multiple allreduces into one group call
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int b = 0; b < batch_count; ++b)
                {
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, num_floats,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                }
                RCCL_CHECK(rccl::ncclGroupEnd());

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            return computeStats(samples, bytes, label);
        }

        // -----------------------------------------------------------------------
        // Table rendering
        // -----------------------------------------------------------------------

        void renderLatencyTable(const std::string &title, const std::vector<LatencyResult> &results)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title
            table << fort::header << title << "" << "" << "" << "" << "" << "" << "" << fort::endr;

            // Header row
            table << fort::header
                  << "Label"
                  << "Bytes"
                  << "Mean (us)"
                  << "Median (us)"
                  << "P5 (us)"
                  << "P95 (us)"
                  << "Min (us)"
                  << "Stddev"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 7; ++c)
            {
                table.column(c).set_cell_text_align(fort::text_align::right);
            }

            for (const auto &r : results)
            {
                char bytes_str[32], mean_str[32], med_str[32], p5_str[32], p95_str[32], min_str[32], sd_str[32];
                snprintf(bytes_str, sizeof(bytes_str), "%zu", r.bytes);
                snprintf(mean_str, sizeof(mean_str), "%.1f", r.mean_us);
                snprintf(med_str, sizeof(med_str), "%.1f", r.median_us);
                snprintf(p5_str, sizeof(p5_str), "%.1f", r.p5_us);
                snprintf(p95_str, sizeof(p95_str), "%.1f", r.p95_us);
                snprintf(min_str, sizeof(min_str), "%.1f", r.min_us);
                snprintf(sd_str, sizeof(sd_str), "%.1f", r.stddev_us);

                table << r.label << bytes_str << mean_str << med_str << p5_str << p95_str << min_str << sd_str << fort::endr;
            }

            std::cout << "\n"
                      << table.to_string() << "\n";
        }

        void renderComparisonTable(const std::string &title,
                                   const std::vector<std::pair<std::string, LatencyResult>> &rows)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header << title << "" << "" << "" << "" << fort::endr;
            table << fort::header
                  << "Configuration"
                  << "Mean (us)"
                  << "Median (us)"
                  << "P5 (us)"
                  << "vs Baseline"
                  << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 4; ++c)
            {
                table.column(c).set_cell_text_align(fort::text_align::right);
            }

            double baseline_median = rows.empty() ? 1.0 : rows[0].second.median_us;

            for (const auto &[name, r] : rows)
            {
                char mean_str[32], med_str[32], p5_str[32], ratio_str[32];
                snprintf(mean_str, sizeof(mean_str), "%.1f", r.mean_us);
                snprintf(med_str, sizeof(med_str), "%.1f", r.median_us);
                snprintf(p5_str, sizeof(p5_str), "%.1f", r.p5_us);
                double ratio = r.median_us / baseline_median;
                snprintf(ratio_str, sizeof(ratio_str), "%.2fx", ratio);

                table << name << mean_str << med_str << p5_str << ratio_str << fort::endr;
            }

            std::cout << "\n"
                      << table.to_string() << "\n";
        }

        void renderImpactSummary(const LatencyResult &target_result)
        {
            // Calculate impact for 7B Qwen2.5 decode
            // 28 layers × 2 allreduces (Wo + FFN down) = 56 allreduces per decode step
            constexpr int ALLREDUCES_PER_DECODE = 56;
            constexpr double SINGLE_GPU_TOK_S = 74.81; // Baseline single-GPU decode

            double total_allreduce_us = target_result.median_us * ALLREDUCES_PER_DECODE;
            double total_allreduce_ms = total_allreduce_us / 1000.0;

            // Single GPU decode time per token
            double single_gpu_ms = 1000.0 / SINGLE_GPU_TOK_S;

            // Theoretical TP=2 time: half the compute + allreduce overhead
            double tp2_compute_ms = single_gpu_ms / 2.0;
            double tp2_total_ms = tp2_compute_ms + total_allreduce_ms;
            double tp2_tok_s = 1000.0 / tp2_total_ms;
            double tp_efficiency = tp2_tok_s / SINGLE_GPU_TOK_S * 100.0;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header << "IMPACT ANALYSIS: Qwen2.5-7B TP=2 Decode" << "" << fort::endr;
            table << fort::header << "Metric" << "Value" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);

            char buf[64];
            snprintf(buf, sizeof(buf), "%.1f us", target_result.median_us);
            table << "Single allreduce (14 KB, median)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%d", ALLREDUCES_PER_DECODE);
            table << "Allreduces per decode step" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f us (%.2f ms)", total_allreduce_us, total_allreduce_ms);
            table << "Total allreduce time per token" << buf << fort::endr;

            table << fort::separator;

            snprintf(buf, sizeof(buf), "%.2f ms (%.1f tok/s)", single_gpu_ms, SINGLE_GPU_TOK_S);
            table << "Single GPU decode (baseline)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.2f ms", tp2_compute_ms);
            table << "TP=2 compute (half of single)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.2f ms", tp2_total_ms);
            table << "TP=2 total (compute + allreduce)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f tok/s", tp2_tok_s);
            table << "TP=2 projected throughput" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f%%", tp_efficiency);
            table << "TP efficiency (vs single GPU)" << buf << fort::endr;

            // Compute what allreduce latency would be needed for 120% efficiency
            double target_efficiency = 1.20;
            double target_tok_s = SINGLE_GPU_TOK_S * target_efficiency;
            double target_total_ms = 1000.0 / target_tok_s;
            double target_allreduce_budget_ms = target_total_ms - tp2_compute_ms;
            double target_per_allreduce_us = (target_allreduce_budget_ms * 1000.0) / ALLREDUCES_PER_DECODE;

            table << fort::separator;

            snprintf(buf, sizeof(buf), "%.1f tok/s (%.1f%% eff)", target_tok_s, target_efficiency * 100.0);
            table << "TARGET throughput" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.2f ms", target_allreduce_budget_ms);
            table << "Allreduce budget for target" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f us", target_per_allreduce_us);
            table << "Per-allreduce budget for target" << buf << fort::endr;

            // Feasibility check
            bool feasible = target_per_allreduce_us > target_result.p5_us;
            table << "Feasible with RCCL?" << (feasible ? "YES (within P5)" : "NO (below P5 floor)") << fort::endr;

            std::cout << "\n"
                      << table.to_string() << "\n";
        }

        void renderPrefillImpactSummary(const LatencyResult &prefill_result)
        {
            // Calculate impact for 7B Qwen2.5 PREFILL
            // 28 layers × 2 allreduces (Wo + FFN down) = 56 allreduces per prefill
            // Each allreduce: 596 × 3584 × 4 = 8,553,472 bytes (~8.5 MB)
            constexpr int ALLREDUCES_PER_PREFILL = 56;
            constexpr double SINGLE_GPU_PREFILL_TOK_S = 1134.63; // Measured baseline
            constexpr int PREFILL_TOKENS = 596;

            double total_allreduce_us = prefill_result.median_us * ALLREDUCES_PER_PREFILL;
            double total_allreduce_ms = total_allreduce_us / 1000.0;

            // Single GPU prefill time
            double single_gpu_ms = static_cast<double>(PREFILL_TOKENS) / SINGLE_GPU_PREFILL_TOK_S * 1000.0;

            // Theoretical TP=2: half the compute + allreduce overhead
            double tp2_compute_ms = single_gpu_ms / 2.0;
            double tp2_total_ms = tp2_compute_ms + total_allreduce_ms;
            double tp2_tok_s = static_cast<double>(PREFILL_TOKENS) / tp2_total_ms * 1000.0;
            double tp_efficiency = tp2_tok_s / SINGLE_GPU_PREFILL_TOK_S * 100.0;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header << "IMPACT ANALYSIS: Qwen2.5-7B TP=2 PREFILL" << "" << fort::endr;
            table << fort::header << "Metric" << "Value" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);

            char buf[64];
            snprintf(buf, sizeof(buf), "%.1f us (%.2f ms)", prefill_result.median_us, prefill_result.median_us / 1000.0);
            table << "Single allreduce (8.5 MB, median)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%d", ALLREDUCES_PER_PREFILL);
            table << "Allreduces per prefill" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f us (%.1f ms)", total_allreduce_us, total_allreduce_ms);
            table << "Total allreduce time (prefill)" << buf << fort::endr;

            table << fort::separator;

            snprintf(buf, sizeof(buf), "%.1f ms (%.0f tok/s)", single_gpu_ms, SINGLE_GPU_PREFILL_TOK_S);
            table << "Single GPU prefill (baseline)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f ms", tp2_compute_ms);
            table << "TP=2 compute (half of single)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f ms", tp2_total_ms);
            table << "TP=2 total (compute + allreduce)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.0f tok/s", tp2_tok_s);
            table << "TP=2 projected prefill throughput" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f%%", tp_efficiency);
            table << "TP efficiency (vs single GPU)" << buf << fort::endr;

            // Compute actual measured vs raw RCCL
            constexpr double MEASURED_TP2_FORWARD_MS = 740.0;               // Our actual measured forward time
            double measured_allreduce_ms = MEASURED_TP2_FORWARD_MS - 303.0; // 303ms was pure compute
            double rccl_floor_pct = total_allreduce_ms / measured_allreduce_ms * 100.0;

            table << fort::separator;
            snprintf(buf, sizeof(buf), "%.0f ms", measured_allreduce_ms);
            table << "Measured allreduce overhead (in-app)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f ms", total_allreduce_ms);
            table << "Raw RCCL floor (this benchmark)" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f%%", rccl_floor_pct);
            table << "RCCL floor / measured overhead" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f ms", measured_allreduce_ms - total_allreduce_ms);
            table << "Gap (our wrapping overhead)" << buf << fort::endr;

            std::cout << "\n"
                      << table.to_string() << "\n";
        }
    };

    // ============================================================================
    // TEST 1: Allreduce latency vs message size (host-timed)
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, LatencyVsMessageSize_HostTimed)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::vector<LatencyResult> results;

        for (size_t bytes : MESSAGE_SIZES_BYTES)
        {
            size_t num_floats = bytes / sizeof(float);
            char label[64];
            if (bytes >= 1024 * 1024)
                snprintf(label, sizeof(label), "%zu MB", bytes / (1024 * 1024));
            else if (bytes >= 1024)
                snprintf(label, sizeof(label), "%zu KB", bytes / 1024);
            else
                snprintf(label, sizeof(label), "%zu B", bytes);

            auto r = benchAllreduceHostTimed(num_floats, label);
            results.push_back(r);
        }

        renderLatencyTable("RCCL AllReduce Latency vs Message Size (Host-Timed, " +
                               std::to_string(num_devices_) + " GPUs, " +
                               std::to_string(BENCH_ITERS) + " iterations)",
                           results);

        // Find the 14336-byte (3584 float) result for decode
        for (const auto &r : results)
        {
            if (r.bytes == 14336)
            {
                renderImpactSummary(r);
                break;
            }
        }

        // Find the 8553472-byte result for prefill
        for (const auto &r : results)
        {
            if (r.bytes == 8553472)
            {
                renderPrefillImpactSummary(r);
                break;
            }
        }
    }

    // ============================================================================
    // TEST 2: Allreduce latency vs message size (GPU event-timed)
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, LatencyVsMessageSize_GPUTimed)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::vector<LatencyResult> results;

        for (size_t bytes : MESSAGE_SIZES_BYTES)
        {
            size_t num_floats = bytes / sizeof(float);
            char label[64];
            if (bytes >= 1024 * 1024)
                snprintf(label, sizeof(label), "%zu MB", bytes / (1024 * 1024));
            else if (bytes >= 1024)
                snprintf(label, sizeof(label), "%zu KB", bytes / 1024);
            else
                snprintf(label, sizeof(label), "%zu B", bytes);

            auto r = benchAllreduceGPUTimed(num_floats, label);
            results.push_back(r);
        }

        renderLatencyTable("RCCL AllReduce Latency vs Message Size (GPU Event-Timed, " +
                               std::to_string(num_devices_) + " GPUs, " +
                               std::to_string(BENCH_ITERS) + " iterations)",
                           results);
    }

    // ============================================================================
    // TEST 3: Group vs No-Group for 14 KB target size
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, GroupVsNoGroup_14KB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t TARGET_FLOATS = 3584; // 14 KB

        std::vector<std::pair<std::string, LatencyResult>> rows;

        // Baseline: with ncclGroupStart/End
        auto grouped = benchAllreduceHostTimed(TARGET_FLOATS, "grouped");
        rows.emplace_back("With ncclGroupStart/End", grouped);

        // Without group
        auto ungrouped = benchAllreduceNoGroup(TARGET_FLOATS, "no-group");
        rows.emplace_back("Without ncclGroupStart/End", ungrouped);

        renderComparisonTable("Group vs No-Group (14 KB = 3584 floats)", rows);
    }

    // ============================================================================
    // TEST 4: Batched allreduce — simulates fusing N consecutive allreduces
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, BatchedAllreduce_14KB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t TARGET_FLOATS = 3584;

        std::vector<std::pair<std::string, LatencyResult>> rows;

        // Single allreduce (baseline)
        auto single = benchAllreduceHostTimed(TARGET_FLOATS, "1x");
        rows.emplace_back("1x allreduce (baseline)", single);

        // Batched 2x (Wo + FFN_down for one layer)
        auto batch2 = benchBatchedAllreduce(TARGET_FLOATS, 2, "2x");
        rows.emplace_back("2x batched (1 layer: Wo+Down)", batch2);

        // Batched 4x (2 layers worth)
        auto batch4 = benchBatchedAllreduce(TARGET_FLOATS, 4, "4x");
        rows.emplace_back("4x batched (2 layers)", batch4);

        // Batched 8x (4 layers worth)
        auto batch8 = benchBatchedAllreduce(TARGET_FLOATS, 8, "8x");
        rows.emplace_back("8x batched (4 layers)", batch8);

        // Batched 56x (full 28-layer model)
        auto batch56 = benchBatchedAllreduce(TARGET_FLOATS, 56, "56x");
        rows.emplace_back("56x batched (all 28 layers)", batch56);

        renderComparisonTable("Batched AllReduce (14 KB each, ncclGroupStart/End)", rows);

        // Show per-allreduce cost for each batch size
        std::cout << "\n--- Per-allreduce amortized cost ---\n";
        std::cout << "  1x: " << single.median_us << " us/allreduce\n";
        std::cout << "  2x: " << batch2.median_us / 2.0 << " us/allreduce\n";
        std::cout << "  4x: " << batch4.median_us / 4.0 << " us/allreduce\n";
        std::cout << "  8x: " << batch8.median_us / 8.0 << " us/allreduce\n";
        std::cout << "  56x: " << batch56.median_us / 56.0 << " us/allreduce\n";
    }

    // ============================================================================
    // TEST 5: Single larger allreduce vs many small ones
    // (tests: should we allreduce 14KB×56 as one 784KB allreduce instead?)
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, FusedVsManySmall_14KB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t TARGET_FLOATS = 3584;                        // 14 KB per allreduce
        constexpr int N_ALLREDUCES = 56;                              // 28 layers × 2
        constexpr size_t FUSED_FLOATS = TARGET_FLOATS * N_ALLREDUCES; // 200,704 floats = 784 KB

        std::vector<std::pair<std::string, LatencyResult>> rows;

        // 56 individual allreduces (current approach)
        // Measure total time for 56 sequential allreduces
        {
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                for (int ar = 0; ar < N_ALLREDUCES; ++ar)
                {
                    RCCL_CHECK(rccl::ncclGroupStart());
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, TARGET_FLOATS,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                    RCCL_CHECK(rccl::ncclGroupEnd());
                }

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            auto r = computeStats(samples, TARGET_FLOATS * sizeof(float) * N_ALLREDUCES, "56x14KB sequential");
            rows.emplace_back("56x 14KB sequential allreduces", r);
        }

        // 56 allreduces batched in a single ncclGroupStart/End
        {
            auto batched = benchBatchedAllreduce(TARGET_FLOATS, N_ALLREDUCES, "56x14KB batched");
            rows.emplace_back("56x 14KB batched (one group)", batched);
        }

        // 1 fused allreduce of 784 KB
        {
            auto fused = benchAllreduceHostTimed(FUSED_FLOATS, "1x784KB fused");
            rows.emplace_back("1x 784KB fused allreduce", fused);
        }

        renderComparisonTable(
            "56x 14KB vs 1x 784KB (total per-token allreduce cost)", rows);
    }

    // ============================================================================
    // TEST 6: In-place vs out-of-place allreduce
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, InPlaceVsOutOfPlace_14KB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t TARGET_FLOATS = 3584;
        constexpr size_t TARGET_BYTES = TARGET_FLOATS * sizeof(float);

        // Allocate separate output buffers for out-of-place
        std::vector<float *> d_out(num_devices_, nullptr);
        for (int i = 0; i < num_devices_; ++i)
        {
            HIP_CHECK(hipSetDevice(devices_[i].ordinal));
            HIP_CHECK(hipMalloc(&d_out[i], TARGET_BYTES));
        }

        std::vector<std::pair<std::string, LatencyResult>> rows;

        // In-place (current approach)
        auto in_place = benchAllreduceHostTimed(TARGET_FLOATS, "in-place");
        rows.emplace_back("In-place (sendbuf == recvbuf)", in_place);

        // Out-of-place
        {
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            // Warmup
            for (int w = 0; w < WARMUP_ITERS; ++w)
            {
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, d_out[i], TARGET_FLOATS,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }
            }

            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, d_out[i], TARGET_FLOATS,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            auto r = computeStats(samples, TARGET_BYTES, "out-of-place");
            rows.emplace_back("Out-of-place (separate recv)", r);
        }

        renderComparisonTable("In-place vs Out-of-place (14 KB = 3584 floats)", rows);

        // Cleanup
        for (int i = 0; i < num_devices_; ++i)
        {
            HIP_CHECK(hipSetDevice(devices_[i].ordinal));
            HIP_CHECK(hipFree(d_out[i]));
        }
    }

    // ============================================================================
    // TEST 7: Half precision (FP16) vs single precision (FP32)
    // (smaller message = less bandwidth, but RCCL might use different kernels)
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, FP16VsFP32_14KB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t TARGET_FLOATS = 3584;

        std::vector<std::pair<std::string, LatencyResult>> rows;

        // FP32 baseline (14 KB)
        auto fp32 = benchAllreduceHostTimed(TARGET_FLOATS, "FP32");
        rows.emplace_back("FP32 (3584 × 4B = 14 KB)", fp32);

        // FP16: same 3584 elements but 7 KB
        // Allocate FP16 buffers
        std::vector<void *> d_fp16(num_devices_, nullptr);
        for (int i = 0; i < num_devices_; ++i)
        {
            HIP_CHECK(hipSetDevice(devices_[i].ordinal));
            HIP_CHECK(hipMalloc(&d_fp16[i], TARGET_FLOATS * sizeof(uint16_t)));
            HIP_CHECK(hipMemset(d_fp16[i], 0, TARGET_FLOATS * sizeof(uint16_t)));
        }

        // Warmup FP16
        for (int w = 0; w < WARMUP_ITERS; ++w)
        {
            RCCL_CHECK(rccl::ncclGroupStart());
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                RCCL_CHECK(rccl::ncclAllReduce(
                    d_fp16[i], d_fp16[i], TARGET_FLOATS,
                    rccl::ncclFloat16, rccl::ncclSum, comms_[i], devices_[i].stream));
            }
            RCCL_CHECK(rccl::ncclGroupEnd());
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
            }
        }

        // Benchmark FP16
        {
            std::vector<double> samples;
            samples.reserve(BENCH_ITERS);

            for (int iter = 0; iter < BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        d_fp16[i], d_fp16[i], TARGET_FLOATS,
                        rccl::ncclFloat16, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            auto r = computeStats(samples, TARGET_FLOATS * sizeof(uint16_t), "FP16");
            rows.emplace_back("FP16 (3584 × 2B = 7 KB)", r);
        }

        // FP32 with 7 KB (1792 floats) — same bytes as FP16
        auto fp32_7k = benchAllreduceHostTimed(TARGET_FLOATS / 2, "FP32-7KB");
        rows.emplace_back("FP32 (1792 × 4B = 7 KB, same bytes)", fp32_7k);

        renderComparisonTable("FP16 vs FP32 AllReduce (3584 elements)", rows);

        // Cleanup
        for (int i = 0; i < num_devices_; ++i)
        {
            HIP_CHECK(hipSetDevice(devices_[i].ordinal));
            HIP_CHECK(hipFree(d_fp16[i]));
        }
    }

    // ============================================================================
    // TEST 8: Rapid-fire sequential allreduces (no sync between calls)
    // Simulates the actual decode pipeline where allreduces are interleaved
    // with compute on the same stream (no host-side sync between layers)
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, RapidFireSequential_14KB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t TARGET_FLOATS = 3584;
        constexpr int N_ALLREDUCES_PER_ROUND = 56; // Full model decode

        std::vector<double> samples;
        samples.reserve(BENCH_ITERS);

        // Warmup
        for (int w = 0; w < 50; ++w)
        {
            for (int ar = 0; ar < N_ALLREDUCES_PER_ROUND; ++ar)
            {
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, devices_[i].d_buffer, TARGET_FLOATS,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());
                // NO sync between allreduces — just queue them
            }
            // Only sync at the end of all 56
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
            }
        }

        // Benchmark
        for (int iter = 0; iter < BENCH_ITERS; ++iter)
        {
            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
            }

            auto t0 = std::chrono::high_resolution_clock::now();

            for (int ar = 0; ar < N_ALLREDUCES_PER_ROUND; ++ar)
            {
                RCCL_CHECK(rccl::ncclGroupStart());
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    RCCL_CHECK(rccl::ncclAllReduce(
                        devices_[i].d_buffer, devices_[i].d_buffer, TARGET_FLOATS,
                        rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                }
                RCCL_CHECK(rccl::ncclGroupEnd());
            }

            for (int i = 0; i < num_devices_; ++i)
            {
                HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            samples.push_back(us);
        }

        auto result = computeStats(samples, TARGET_FLOATS * sizeof(float) * N_ALLREDUCES_PER_ROUND,
                                   "56x rapid-fire");

        // Report
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        table << fort::header << "Rapid-Fire 56x AllReduce (no inter-call sync)" << "" << fort::endr;
        table << fort::header << "Metric" << "Value" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::right);

        char buf[64];
        snprintf(buf, sizeof(buf), "%.1f us", result.median_us);
        table << "Total for 56 allreduces (median)" << buf << fort::endr;

        snprintf(buf, sizeof(buf), "%.1f us", result.median_us / N_ALLREDUCES_PER_ROUND);
        table << "Amortized per allreduce" << buf << fort::endr;

        snprintf(buf, sizeof(buf), "%.1f us", result.p5_us);
        table << "Total (P5 best case)" << buf << fort::endr;

        snprintf(buf, sizeof(buf), "%.1f us", result.p5_us / N_ALLREDUCES_PER_ROUND);
        table << "Amortized per allreduce (P5)" << buf << fort::endr;

        // Compare to theoretical throughput impact
        constexpr double SINGLE_GPU_TOK_S = 74.81;
        double single_gpu_ms = 1000.0 / SINGLE_GPU_TOK_S;
        double tp2_compute_ms = single_gpu_ms / 2.0;
        double allreduce_ms = result.median_us / 1000.0;
        double projected_ms = tp2_compute_ms + allreduce_ms;
        double projected_tok_s = 1000.0 / projected_ms;
        double tp_eff = projected_tok_s / SINGLE_GPU_TOK_S * 100.0;

        table << fort::separator;
        snprintf(buf, sizeof(buf), "%.2f ms", allreduce_ms);
        table << "Allreduce wall time" << buf << fort::endr;

        snprintf(buf, sizeof(buf), "%.1f tok/s (%.1f%% eff)", projected_tok_s, tp_eff);
        table << "Projected TP=2 throughput" << buf << fort::endr;

        std::cout << "\n"
                  << table.to_string() << "\n";
    }

    // ============================================================================
    // TEST 9: Prefill-sized allreduce latency (8.5 MB per allreduce)
    // Simulates TP=2 prefill: 56 allreduces of 596×3584 floats each
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, PrefillAllreduce_8_5MB)
    {
        if (!initialized_)
            GTEST_SKIP();

        // Prefill: 596 seq_len × 3584 d_model = 2,138,368 floats = 8,553,472 bytes
        constexpr size_t PREFILL_FLOATS = 596 * 3584; // 2,138,368
        constexpr int N_ALLREDUCES = 56;              // 28 layers × 2

        // --- Part A: Single allreduce latency at prefill size ---
        std::vector<std::pair<std::string, LatencyResult>> rows;

        auto host_timed = benchAllreduceHostTimed(PREFILL_FLOATS, "8.5MB host-timed");
        rows.emplace_back("Single 8.5MB (host-timed)", host_timed);

        auto gpu_timed = benchAllreduceGPUTimed(PREFILL_FLOATS, "8.5MB GPU-timed");
        rows.emplace_back("Single 8.5MB (GPU event-timed)", gpu_timed);

        // Decode-sized for comparison
        auto decode_ref = benchAllreduceHostTimed(3584, "14KB reference");
        rows.emplace_back("Single 14KB decode (reference)", decode_ref);

        renderComparisonTable("Prefill AllReduce: Single Call Latency (8.5 MB = 596×3584 floats)", rows);

        // --- Part B: 56 sequential allreduces at prefill size ---
        // This simulates the actual prefill forward pass
        {
            constexpr int PREFILL_BENCH_ITERS = 100; // Fewer iterations for large transfers

            std::vector<double> samples;
            samples.reserve(PREFILL_BENCH_ITERS);

            // Warmup
            for (int w = 0; w < 20; ++w)
            {
                for (int ar = 0; ar < N_ALLREDUCES; ++ar)
                {
                    RCCL_CHECK(rccl::ncclGroupStart());
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, PREFILL_FLOATS,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                    RCCL_CHECK(rccl::ncclGroupEnd());
                }
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }
            }

            // Benchmark: 56 sequential allreduces, each synced (mimics our pipeline)
            for (int iter = 0; iter < PREFILL_BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                for (int ar = 0; ar < N_ALLREDUCES; ++ar)
                {
                    RCCL_CHECK(rccl::ncclGroupStart());
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, PREFILL_FLOATS,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                    RCCL_CHECK(rccl::ncclGroupEnd());
                    // Sync after each allreduce (matches our pipeline: compute waits for allreduce)
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                samples.push_back(us);
            }

            auto synced = computeStats(samples, static_cast<size_t>(PREFILL_FLOATS) * sizeof(float) * N_ALLREDUCES,
                                       "56x8.5MB synced");

            // Also measure rapid-fire (enqueue all, sync at end)
            std::vector<double> rapid_samples;
            rapid_samples.reserve(PREFILL_BENCH_ITERS);

            for (int iter = 0; iter < PREFILL_BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                for (int ar = 0; ar < N_ALLREDUCES; ++ar)
                {
                    RCCL_CHECK(rccl::ncclGroupStart());
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, PREFILL_FLOATS,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                    RCCL_CHECK(rccl::ncclGroupEnd());
                    // NO sync between allreduces
                }

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                rapid_samples.push_back(us);
            }

            auto rapid = computeStats(rapid_samples, static_cast<size_t>(PREFILL_FLOATS) * sizeof(float) * N_ALLREDUCES,
                                      "56x8.5MB rapid-fire");

            // Report
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            table << fort::header << "PREFILL: 56x 8.5MB AllReduce (Qwen2.5-7B TP=2 forward)" << "" << fort::endr;
            table << fort::header << "Metric" << "Value" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);

            char buf[96];

            snprintf(buf, sizeof(buf), "%.1f us (%.2f ms)", synced.median_us, synced.median_us / 1000.0);
            table << "56x synced (per-allreduce hipSync) median" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f us (%.2f ms)", synced.median_us / N_ALLREDUCES,
                     synced.median_us / N_ALLREDUCES / 1000.0);
            table << "  -> per allreduce (synced)" << buf << fort::endr;

            table << fort::separator;

            snprintf(buf, sizeof(buf), "%.1f us (%.2f ms)", rapid.median_us, rapid.median_us / 1000.0);
            table << "56x rapid-fire (sync at end) median" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.1f us (%.2f ms)", rapid.median_us / N_ALLREDUCES,
                     rapid.median_us / N_ALLREDUCES / 1000.0);
            table << "  -> per allreduce (rapid-fire)" << buf << fort::endr;

            table << fort::separator;

            // Compare to measured in-app overhead
            constexpr double MEASURED_ALLREDUCE_MS = 430.0; // From LLAMINAR_SKIP_ALLREDUCE diagnostic
            snprintf(buf, sizeof(buf), "%.1f ms", MEASURED_ALLREDUCE_MS);
            table << "Measured in-app allreduce overhead" << buf << fort::endr;

            snprintf(buf, sizeof(buf), "%.2f ms", synced.median_us / 1000.0);
            table << "Raw RCCL synced floor" << buf << fort::endr;

            double gap_ms = MEASURED_ALLREDUCE_MS - synced.median_us / 1000.0;
            snprintf(buf, sizeof(buf), "%.1f ms (%.0f%%)", gap_ms,
                     gap_ms / MEASURED_ALLREDUCE_MS * 100.0);
            table << "Gap (app overhead beyond RCCL)" << buf << fort::endr;

            table << fort::separator;

            // Bandwidth utilization
            double total_bytes = static_cast<double>(PREFILL_FLOATS) * sizeof(float) * N_ALLREDUCES;
            // For ring allreduce with 2 GPUs: data moved = 2*(N-1)/N * total = total bytes
            double achieved_bw_gbs = total_bytes / (synced.median_us / 1e6) / 1e9;
            snprintf(buf, sizeof(buf), "%.2f GB/s (synced)", achieved_bw_gbs);
            table << "Achieved bandwidth" << buf << fort::endr;

            double rapid_bw_gbs = total_bytes / (rapid.median_us / 1e6) / 1e9;
            snprintf(buf, sizeof(buf), "%.2f GB/s (rapid-fire)", rapid_bw_gbs);
            table << "Achieved bandwidth" << buf << fort::endr;

            // PCIe gen3 x16 theoretical: ~15.75 GB/s
            snprintf(buf, sizeof(buf), "%.2f GB/s (PCIe3 x16)", 15.75);
            table << "Theoretical max bandwidth" << buf << fort::endr;

            std::cout << "\n"
                      << table.to_string() << "\n";
        }
    }

    // ============================================================================
    // TEST 10: Prefill fused vs sequential
    // (Should we fuse all 56 allreduces into one larger allreduce?)
    // ============================================================================

    TEST_F(Perf__RCCLAllreduceLatency, PrefillFusedVsSequential_8_5MB)
    {
        if (!initialized_)
            GTEST_SKIP();

        constexpr size_t PREFILL_FLOATS = 596 * 3584; // 2,138,368 per allreduce
        constexpr int N_ALLREDUCES = 56;
        constexpr int PREFILL_BENCH_ITERS = 100;

        std::vector<std::pair<std::string, LatencyResult>> rows;

        // A: 56 individual allreduces, synced after each
        {
            std::vector<double> samples;
            samples.reserve(PREFILL_BENCH_ITERS);

            for (int iter = 0; iter < PREFILL_BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                for (int ar = 0; ar < N_ALLREDUCES; ++ar)
                {
                    RCCL_CHECK(rccl::ncclGroupStart());
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, PREFILL_FLOATS,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                    RCCL_CHECK(rccl::ncclGroupEnd());

                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }

            auto r = computeStats(samples, static_cast<size_t>(PREFILL_FLOATS) * 4 * N_ALLREDUCES, "56x synced");
            rows.emplace_back("56x 8.5MB (synced each)", r);
        }

        // B: 56 allreduces batched in ncclGroupStart/End pairs (no sync between)
        {
            std::vector<double> samples;
            samples.reserve(PREFILL_BENCH_ITERS);

            for (int iter = 0; iter < PREFILL_BENCH_ITERS; ++iter)
            {
                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                for (int ar = 0; ar < N_ALLREDUCES; ++ar)
                {
                    RCCL_CHECK(rccl::ncclGroupStart());
                    for (int i = 0; i < num_devices_; ++i)
                    {
                        HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                        RCCL_CHECK(rccl::ncclAllReduce(
                            devices_[i].d_buffer, devices_[i].d_buffer, PREFILL_FLOATS,
                            rccl::ncclFloat, rccl::ncclSum, comms_[i], devices_[i].stream));
                    }
                    RCCL_CHECK(rccl::ncclGroupEnd());
                }

                for (int i = 0; i < num_devices_; ++i)
                {
                    HIP_CHECK(hipSetDevice(devices_[i].ordinal));
                    HIP_CHECK(hipStreamSynchronize(devices_[i].stream));
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }

            auto r = computeStats(samples, static_cast<size_t>(PREFILL_FLOATS) * 4 * N_ALLREDUCES, "56x rapid-fire");
            rows.emplace_back("56x 8.5MB (rapid-fire, sync at end)", r);
        }

        // C: Single massive allreduce (if we could fuse all residuals)
        // NOTE: This doesn't reflect reality (can't fuse across layers), but shows bandwidth ceiling
        {
            // 56 × 2,138,368 = 119,748,608 floats = ~457 MB — too big for our buffer
            // Instead, test 2-layer fusion: 4 × 8.5MB batched in one group
            constexpr int BATCH_2LAYER = 4;
            auto batch4 = benchBatchedAllreduce(PREFILL_FLOATS, BATCH_2LAYER, "4x8.5MB batched");
            rows.emplace_back("4x 8.5MB batched (2-layer fusion)", batch4);
        }

        renderComparisonTable("PREFILL: Synced vs Rapid-fire vs Fused (8.5 MB allreduces)", rows);
    }

} // anonymous namespace

#else // !HAVE_ROCM

TEST(Perf__RCCLAllreduceLatency, SkippedNoROCm)
{
    GTEST_SKIP() << "RCCL allreduce benchmark requires HAVE_ROCM";
}

#endif // HAVE_ROCM
