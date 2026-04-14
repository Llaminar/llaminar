/**
 * @file Perf__HostBackendCrossVendorCopy.cpp
 * @brief Bandwidth and latency benchmark for HostBackend cross-vendor GPU copies
 *
 * Measures HostBackend::copy() performance for CUDA↔ROCm transfers through
 * the pre-allocated pinned staging buffer. Reports:
 * - Latency (μs) per copy at various sizes
 * - Bandwidth (GB/s) at various sizes
 * - Breakdown: D2H leg, H2D leg, and total
 *
 * Requires both CUDA and ROCm devices to be present.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <vector>

#include "collective/backends/HostBackend.h"
#include "collective/DeviceGroup.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"

// libfort for tables
#include "fort.hpp"

using namespace llaminar2;

// ============================================================================
// Configuration
// ============================================================================

namespace
{
    constexpr int WARMUP_ITERS = 5;
    constexpr int BENCH_ITERS = 20;

    // Transfer sizes to benchmark
    const std::vector<size_t> TRANSFER_SIZES = {
        4 * 1024,          //   4 KB
        64 * 1024,         //  64 KB
        256 * 1024,        // 256 KB
        1024 * 1024,       //   1 MB
        4 * 1024 * 1024,   //   4 MB
        16 * 1024 * 1024,  //  16 MB
        64 * 1024 * 1024,  //  64 MB
        128 * 1024 * 1024, // 128 MB
    };

    std::string bytesLabel(size_t bytes)
    {
        if (bytes >= 1024ULL * 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
        if (bytes >= 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        if (bytes >= 1024)
            return std::to_string(bytes / 1024) + " KB";
        return std::to_string(bytes) + " B";
    }

    double toMicroseconds(std::chrono::high_resolution_clock::duration d)
    {
        return std::chrono::duration<double, std::micro>(d).count();
    }

    double bandwidthGBs(size_t bytes, double us)
    {
        if (us <= 0.0)
            return 0.0;
        return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (us / 1e6);
    }
} // namespace

// ============================================================================
// Test fixture
// ============================================================================

class HostBackendCrossVendorBench : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cuda_backend_ = getCUDABackend();
        rocm_backend_ = getROCmBackend();

        if (!cuda_backend_ || !rocm_backend_)
        {
            GTEST_SKIP() << "Requires both CUDA and ROCm backends";
        }

        cuda_dev_ = DeviceId::cuda(0);
        rocm_dev_ = DeviceId::rocm(0);

        // Initialize HostBackend with a group containing both devices
        DeviceGroup group;
        group.name = "bench_cross_vendor";
        group.devices = {cuda_dev_, rocm_dev_};
        ASSERT_TRUE(host_backend_.initialize(group));
    }

    void TearDown() override
    {
        // Free any allocated GPU buffers
        for (auto &[ptr, dev] : allocated_)
        {
            IBackend *be = getBackendFor(dev);
            if (be)
                be->free(ptr, dev.ordinal);
        }
        allocated_.clear();
        host_backend_.shutdown();
    }

    void *allocGPU(DeviceId dev, size_t bytes)
    {
        IBackend *be = getBackendFor(dev);
        EXPECT_NE(be, nullptr);
        void *ptr = be->allocate(bytes, dev.ordinal);
        EXPECT_NE(ptr, nullptr);
        if (ptr)
            allocated_.push_back({ptr, dev});
        return ptr;
    }

    IBackend *cuda_backend_ = nullptr;
    IBackend *rocm_backend_ = nullptr;
    DeviceId cuda_dev_;
    DeviceId rocm_dev_;
    HostBackend host_backend_;
    std::vector<std::pair<void *, DeviceId>> allocated_;
};

// ============================================================================
// ROCm → CUDA benchmark
// ============================================================================

TEST_F(HostBackendCrossVendorBench, ROCmToCUDA_Bandwidth)
{
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header
          << "Size" << "Avg (μs)" << "Min (μs)" << "Max (μs)" << "BW (GB/s)" << fort::endr;

    for (size_t bytes : TRANSFER_SIZES)
    {
        void *src = allocGPU(rocm_dev_, bytes);
        void *dst = allocGPU(cuda_dev_, bytes);
        ASSERT_NE(src, nullptr);
        ASSERT_NE(dst, nullptr);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i)
        {
            ASSERT_TRUE(host_backend_.copy(dst, cuda_dev_, src, rocm_dev_, bytes));
        }

        // Benchmark
        std::vector<double> timings;
        timings.reserve(BENCH_ITERS);

        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            ASSERT_TRUE(host_backend_.copy(dst, cuda_dev_, src, rocm_dev_, bytes));
            auto t1 = std::chrono::high_resolution_clock::now();
            timings.push_back(toMicroseconds(t1 - t0));
        }

        double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        double mn = *std::min_element(timings.begin(), timings.end());
        double mx = *std::max_element(timings.begin(), timings.end());
        double bw = bandwidthGBs(bytes, avg);

        char row_avg[32], row_min[32], row_max[32], row_bw[32];
        snprintf(row_avg, sizeof(row_avg), "%.1f", avg);
        snprintf(row_min, sizeof(row_min), "%.1f", mn);
        snprintf(row_max, sizeof(row_max), "%.1f", mx);
        snprintf(row_bw, sizeof(row_bw), "%.2f", bw);

        table << bytesLabel(bytes) << row_avg << row_min << row_max << row_bw << fort::endr;
    }

    std::cout << "\n╔══════════════════════════════════════════════════╗\n"
              << "║       ROCm:0 → CUDA:0 (HostBackend::copy)       ║\n"
              << "╚══════════════════════════════════════════════════╝\n"
              << table.to_string() << std::endl;
}

// ============================================================================
// CUDA → ROCm benchmark
// ============================================================================

TEST_F(HostBackendCrossVendorBench, CUDAToROCm_Bandwidth)
{
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header
          << "Size" << "Avg (μs)" << "Min (μs)" << "Max (μs)" << "BW (GB/s)" << fort::endr;

    for (size_t bytes : TRANSFER_SIZES)
    {
        void *src = allocGPU(cuda_dev_, bytes);
        void *dst = allocGPU(rocm_dev_, bytes);
        ASSERT_NE(src, nullptr);
        ASSERT_NE(dst, nullptr);

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i)
        {
            ASSERT_TRUE(host_backend_.copy(dst, rocm_dev_, src, cuda_dev_, bytes));
        }

        // Benchmark
        std::vector<double> timings;
        timings.reserve(BENCH_ITERS);

        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            ASSERT_TRUE(host_backend_.copy(dst, rocm_dev_, src, cuda_dev_, bytes));
            auto t1 = std::chrono::high_resolution_clock::now();
            timings.push_back(toMicroseconds(t1 - t0));
        }

        double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        double mn = *std::min_element(timings.begin(), timings.end());
        double mx = *std::max_element(timings.begin(), timings.end());
        double bw = bandwidthGBs(bytes, avg);

        char row_avg[32], row_min[32], row_max[32], row_bw[32];
        snprintf(row_avg, sizeof(row_avg), "%.1f", avg);
        snprintf(row_min, sizeof(row_min), "%.1f", mn);
        snprintf(row_max, sizeof(row_max), "%.1f", mx);
        snprintf(row_bw, sizeof(row_bw), "%.2f", bw);

        table << bytesLabel(bytes) << row_avg << row_min << row_max << row_bw << fort::endr;
    }

    std::cout << "\n╔══════════════════════════════════════════════════╗\n"
              << "║       CUDA:0 → ROCm:0 (HostBackend::copy)       ║\n"
              << "╚══════════════════════════════════════════════════╝\n"
              << table.to_string() << std::endl;
}

// ============================================================================
// Bidirectional (mimics PP decode: one copy per step)
// ============================================================================

TEST_F(HostBackendCrossVendorBench, Bidirectional_PPSimulation)
{
    // Simulate typical PP hidden-state sizes (Qwen2.5 0.5B = 896 dim * seq)
    const std::vector<std::pair<std::string, size_t>> scenarios = {
        {"Decode (1 tok, 896d)", 1 * 896 * sizeof(float)},
        {"Decode (1 tok, 3584d)", 1 * 3584 * sizeof(float)}, // 7B
        {"Prefill (128 tok, 896d)", 128 * 896 * sizeof(float)},
        {"Prefill (128 tok, 3584d)", 128 * 3584 * sizeof(float)}, // 7B
        {"Prefill (512 tok, 896d)", 512 * 896 * sizeof(float)},
        {"Prefill (512 tok, 3584d)", 512 * 3584 * sizeof(float)}, // 7B
    };

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header
          << "Scenario" << "Bytes" << "ROCm→CUDA (μs)" << "CUDA→ROCm (μs)"
          << "BW R→C (GB/s)" << "BW C→R (GB/s)" << fort::endr;

    for (const auto &[name, bytes] : scenarios)
    {
        void *rocm_buf = allocGPU(rocm_dev_, bytes);
        void *cuda_buf = allocGPU(cuda_dev_, bytes);

        // Warmup both directions
        for (int i = 0; i < WARMUP_ITERS; ++i)
        {
            host_backend_.copy(cuda_buf, cuda_dev_, rocm_buf, rocm_dev_, bytes);
            host_backend_.copy(rocm_buf, rocm_dev_, cuda_buf, cuda_dev_, bytes);
        }

        // Benchmark ROCm→CUDA
        std::vector<double> r2c;
        r2c.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            host_backend_.copy(cuda_buf, cuda_dev_, rocm_buf, rocm_dev_, bytes);
            auto t1 = std::chrono::high_resolution_clock::now();
            r2c.push_back(toMicroseconds(t1 - t0));
        }

        // Benchmark CUDA→ROCm
        std::vector<double> c2r;
        c2r.reserve(BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            host_backend_.copy(rocm_buf, rocm_dev_, cuda_buf, cuda_dev_, bytes);
            auto t1 = std::chrono::high_resolution_clock::now();
            c2r.push_back(toMicroseconds(t1 - t0));
        }

        double r2c_avg = std::accumulate(r2c.begin(), r2c.end(), 0.0) / r2c.size();
        double c2r_avg = std::accumulate(c2r.begin(), c2r.end(), 0.0) / c2r.size();

        char s_r2c[32], s_c2r[32], s_bw_r2c[32], s_bw_c2r[32];
        snprintf(s_r2c, sizeof(s_r2c), "%.1f", r2c_avg);
        snprintf(s_c2r, sizeof(s_c2r), "%.1f", c2r_avg);
        snprintf(s_bw_r2c, sizeof(s_bw_r2c), "%.2f", bandwidthGBs(bytes, r2c_avg));
        snprintf(s_bw_c2r, sizeof(s_bw_c2r), "%.2f", bandwidthGBs(bytes, c2r_avg));

        table << name << bytesLabel(bytes) << s_r2c << s_c2r
              << s_bw_r2c << s_bw_c2r << fort::endr;
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"
              << "║       PP Activation Transfer Simulation (HostBackend)       ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n"
              << table.to_string() << std::endl;
}
