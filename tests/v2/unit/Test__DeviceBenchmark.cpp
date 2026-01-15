/**
 * @file Test__DeviceBenchmark.cpp
 * @brief Unit tests for device benchmark infrastructure
 *
 * Tests the DeviceBenchmark interface and CPU benchmark implementation.
 * GPU benchmarks are tested conditionally based on HAVE_CUDA/HAVE_ROCM.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "backends/benchmarks/DeviceBenchmark.h"
#include "backends/benchmarks/CPUBenchmark.h"
#include "utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Test fixture for device benchmarks
 */
class Test__DeviceBenchmark : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use quick config for faster tests
        config_ = BenchmarkConfig::quick();
    }

    BenchmarkConfig config_;
};

// =============================================================================
// BenchmarkConfig Tests
// =============================================================================

TEST_F(Test__DeviceBenchmark, QuickConfigHasReasonableDefaults)
{
    auto config = BenchmarkConfig::quick();

    // Quick config should have smaller memory test
    EXPECT_LE(config.memory_test_bytes, 16 * 1024 * 1024);

    // Quick config should have fewer iterations
    EXPECT_LE(config.iterations, 5);

    // Quick config should disable P2P benchmarks
    EXPECT_FALSE(config.benchmark_p2p);
}

TEST_F(Test__DeviceBenchmark, ThoroughConfigHasLargerValues)
{
    auto config = BenchmarkConfig::thorough();

    // Thorough config should have larger memory test
    EXPECT_GE(config.memory_test_bytes, 32 * 1024 * 1024);

    // Thorough config should have more iterations
    EXPECT_GE(config.iterations, 5);

    // Thorough config should enable P2P benchmarks
    EXPECT_TRUE(config.benchmark_p2p);
}

// =============================================================================
// DeviceBenchmarkFactory Tests
// =============================================================================

TEST_F(Test__DeviceBenchmark, FactoryCreatesCPUBenchmark)
{
    auto benchmark = DeviceBenchmarkFactory::create(DeviceId::cpu(), config_);

    ASSERT_NE(benchmark, nullptr);
    EXPECT_EQ(benchmark->device(), DeviceId::cpu());
}

TEST_F(Test__DeviceBenchmark, EnumerateDevicesIncludesCPU)
{
    auto devices = DeviceBenchmarkFactory::enumerateDevices();

    ASSERT_FALSE(devices.empty());

    // CPU should always be present
    bool has_cpu = false;
    for (const auto &device : devices)
    {
        if (device.type == DeviceType::CPU)
        {
            has_cpu = true;
            break;
        }
    }
    EXPECT_TRUE(has_cpu);
}

// =============================================================================
// CPU Benchmark Tests
// =============================================================================

TEST_F(Test__DeviceBenchmark, CPUBenchmarkRunsSuccessfully)
{
    CPUBenchmark benchmark(config_);

    auto result = benchmark.run();

    // Should complete successfully
    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.isValid());

    // Should have measured some memory bandwidth
    EXPECT_GT(result.memory_read_gbps, 0.0);
    EXPECT_GT(result.memory_write_gbps, 0.0);
    EXPECT_GT(result.memory_copy_gbps, 0.0);

    // Should have measured some compute throughput
    EXPECT_GT(result.compute_fp32_gflops, 0.0);
    EXPECT_GT(result.compute_int8_gops, 0.0);

    // CPU doesn't have transfer rates
    EXPECT_EQ(result.h2d_pinned_gbps, 0.0);
    EXPECT_EQ(result.d2h_pinned_gbps, 0.0);

    // Benchmark should complete in reasonable time
    EXPECT_LT(result.benchmark_duration_ms, 5000.0);
}

TEST_F(Test__DeviceBenchmark, CPUBenchmarkReturnsCorrectDevice)
{
    CPUBenchmark benchmark(config_);

    EXPECT_EQ(benchmark.device(), DeviceId::cpu());

    auto result = benchmark.run();
    EXPECT_EQ(result.device, DeviceId::cpu());
}

TEST_F(Test__DeviceBenchmark, CPUBenchmarkMemoryBandwidthIsReasonable)
{
    CPUBenchmark benchmark(config_);

    auto result = benchmark.run();

    // Memory bandwidth should be in reasonable range for CPUs
    // In virtualized/containerized environments, bandwidth may be lower
    // DDR4: ~25-50 GB/s per channel, DDR5: ~40-80 GB/s per channel
    // With multiple channels and NUMA, could see 100-400 GB/s total
    // But in VMs/containers, we might see much lower (0.1-10 GB/s)
    EXPECT_GT(result.memory_read_gbps, 0.01);   // At least 10 MB/s
    EXPECT_LT(result.memory_read_gbps, 2000.0); // Less than 2 TB/s

    EXPECT_GT(result.memory_write_gbps, 0.01);
    EXPECT_LT(result.memory_write_gbps, 2000.0);

    EXPECT_GT(result.memory_copy_gbps, 0.01);
    EXPECT_LT(result.memory_copy_gbps, 2000.0);
}

TEST_F(Test__DeviceBenchmark, CPUBenchmarkComputeIsReasonable)
{
    CPUBenchmark benchmark(config_);

    auto result = benchmark.run();

    // Just verify we got nonzero results - actual values vary wildly based on:
    // - Debug vs Release build (-O0 vs -O3)
    // - Parallel test execution (resource contention)
    // - Virtualized/containerized environments
    EXPECT_GT(result.compute_fp32_gflops, 0.0);
    EXPECT_LT(result.compute_fp32_gflops, 100000.0); // Sanity upper bound

    EXPECT_GT(result.compute_int8_gops, 0.0);
    EXPECT_LT(result.compute_int8_gops, 500000.0); // Sanity upper bound
}

// =============================================================================
// DeviceBenchmarkRunner Tests
// =============================================================================

TEST_F(Test__DeviceBenchmark, RunnerRunsSingleDevice)
{
    DeviceBenchmarkRunner runner(config_);

    auto result = runner.runSingleDevice(DeviceId::cpu());

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.device, DeviceId::cpu());
}

TEST_F(Test__DeviceBenchmark, RunnerRunsAllDevices)
{
    DeviceBenchmarkRunner runner(config_);

    auto results = runner.runAll();

    // Should have at least CPU results
    EXPECT_FALSE(results.empty());

    // CPU should be in results
    auto it = results.find(DeviceId::cpu());
    EXPECT_NE(it, results.end());
    EXPECT_TRUE(it->second.valid);
}

TEST_F(Test__DeviceBenchmark, RunnerEstimatesDuration)
{
    DeviceBenchmarkRunner runner(config_);

    std::vector<DeviceId> devices = {DeviceId::cpu()};
    double estimate = runner.estimateTotalDuration(devices);

    // Should provide a positive estimate
    EXPECT_GT(estimate, 0.0);
}

TEST_F(Test__DeviceBenchmark, CrossVendorTransferMeasurement)
{
    // This test only runs if both CUDA and ROCm devices are available
    auto devices = DeviceBenchmarkFactory::enumerateDevices();

    bool has_cuda = false;
    bool has_rocm = false;
    DeviceId cuda_dev, rocm_dev;

    for (const auto &dev : devices)
    {
        if (dev.type == DeviceType::CUDA && !has_cuda)
        {
            has_cuda = true;
            cuda_dev = dev;
        }
        else if (dev.type == DeviceType::ROCm && !has_rocm)
        {
            has_rocm = true;
            rocm_dev = dev;
        }
    }

    if (!has_cuda || !has_rocm)
    {
        GTEST_SKIP() << "Requires both CUDA and ROCm devices for cross-vendor test";
    }

    // Enable cross-vendor benchmarks
    BenchmarkConfig config = BenchmarkConfig::quick();
    config.benchmark_cross_vendor = true;

    DeviceBenchmarkRunner runner(config);

    // Measure CUDA -> ROCm
    double cuda_to_rocm = runner.measureCrossVendorTransfer(cuda_dev, rocm_dev);
    EXPECT_GT(cuda_to_rocm, 0.0) << "CUDA -> ROCm transfer should succeed";

    // Measure ROCm -> CUDA
    double rocm_to_cuda = runner.measureCrossVendorTransfer(rocm_dev, cuda_dev);
    EXPECT_GT(rocm_to_cuda, 0.0) << "ROCm -> CUDA transfer should succeed";

    LOG_INFO("Cross-vendor transfer rates:");
    LOG_INFO("  CUDA -> ROCm: " << cuda_to_rocm << " GB/s");
    LOG_INFO("  ROCm -> CUDA: " << rocm_to_cuda << " GB/s");
}

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/benchmarks/CrossVendorP2P.h"

TEST_F(Test__DeviceBenchmark, CrossVendorP2PEngineBasic)
{
    // Find CUDA and ROCm devices
    auto devices = DeviceBenchmarkFactory::enumerateDevices();

    DeviceId cuda_dev, rocm_dev;
    bool has_cuda = false, has_rocm = false;

    for (const auto &dev : devices)
    {
        if (dev.type == DeviceType::CUDA && !has_cuda)
        {
            has_cuda = true;
            cuda_dev = dev;
        }
        else if (dev.type == DeviceType::ROCm && !has_rocm)
        {
            has_rocm = true;
            rocm_dev = dev;
        }
    }

    if (!has_cuda || !has_rocm)
    {
        GTEST_SKIP() << "Requires both CUDA and ROCm devices";
    }

    // Test the pipelined engine
    CrossVendorP2PConfig config;
    config.buffer_size = 16 * 1024 * 1024; // 16 MB buffers
    config.chunk_size = 4 * 1024 * 1024;   // 4 MB chunks
    config.num_buffers = 2;
    config.enable_pipelining = true;

    CrossVendorP2PEngine engine(config);
    ASSERT_TRUE(engine.initialize(cuda_dev, rocm_dev));

    // Benchmark
    auto result = engine.benchmark(64 * 1024 * 1024, 3); // 64 MB, 3 iterations
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.throughput_gbps, 0.0);

    LOG_INFO("CrossVendorP2PEngine benchmark:");
    LOG_INFO("  CUDA -> ROCm pipelined: " << result.throughput_gbps << " GB/s");
}

TEST_F(Test__DeviceBenchmark, CrossVendorP2PPipelinedVsSequential)
{
    // Compare pipelined vs sequential performance
    auto devices = DeviceBenchmarkFactory::enumerateDevices();

    DeviceId cuda_dev, rocm_dev;
    bool has_cuda = false, has_rocm = false;

    for (const auto &dev : devices)
    {
        if (dev.type == DeviceType::CUDA && !has_cuda)
        {
            has_cuda = true;
            cuda_dev = dev;
        }
        else if (dev.type == DeviceType::ROCm && !has_rocm)
        {
            has_rocm = true;
            rocm_dev = dev;
        }
    }

    if (!has_cuda || !has_rocm)
    {
        GTEST_SKIP() << "Requires both CUDA and ROCm devices";
    }

    const size_t test_size = 64 * 1024 * 1024; // 64 MB

    // Sequential (no pipelining)
    CrossVendorP2PConfig seq_config;
    seq_config.buffer_size = 16 * 1024 * 1024;
    seq_config.enable_pipelining = false;
    seq_config.num_buffers = 1;

    CrossVendorP2PEngine seq_engine(seq_config);
    ASSERT_TRUE(seq_engine.initialize(cuda_dev, rocm_dev));
    auto seq_result = seq_engine.benchmark(test_size, 3);
    EXPECT_TRUE(seq_result.success);

    // Pipelined
    CrossVendorP2PConfig pipe_config;
    pipe_config.buffer_size = 16 * 1024 * 1024;
    pipe_config.chunk_size = 4 * 1024 * 1024;
    pipe_config.enable_pipelining = true;
    pipe_config.num_buffers = 2;

    CrossVendorP2PEngine pipe_engine(pipe_config);
    ASSERT_TRUE(pipe_engine.initialize(cuda_dev, rocm_dev));
    auto pipe_result = pipe_engine.benchmark(test_size, 3);
    EXPECT_TRUE(pipe_result.success);

    LOG_INFO("Cross-vendor P2P comparison (CUDA -> ROCm):");
    LOG_INFO("  Sequential:  " << seq_result.throughput_gbps << " GB/s");
    LOG_INFO("  Pipelined:   " << pipe_result.throughput_gbps << " GB/s");
    LOG_INFO("  Speedup:     " << (pipe_result.throughput_gbps / seq_result.throughput_gbps) << "x");

    // Pipelined should be faster (or at least not slower)
    EXPECT_GE(pipe_result.throughput_gbps, seq_result.throughput_gbps * 0.95);
}
#endif // HAVE_CUDA && HAVE_ROCM

// =============================================================================
// DeviceBenchmarkResult Tests
// =============================================================================

TEST_F(Test__DeviceBenchmark, ResultComputesEffectiveBandwidth)
{
    DeviceBenchmarkResult result;
    result.memory_read_gbps = 100.0;
    result.memory_write_gbps = 80.0;
    result.memory_copy_gbps = 120.0;

    // Effective bandwidth: 0.5*copy + 0.3*read + 0.2*write
    double expected = 0.5 * 120.0 + 0.3 * 100.0 + 0.2 * 80.0;
    EXPECT_DOUBLE_EQ(result.memory_bandwidth_gbps(), expected);
}

TEST_F(Test__DeviceBenchmark, ResultChecksP2PAvailability)
{
    DeviceBenchmarkResult result;

    // No P2P initially
    EXPECT_FALSE(result.hasP2P(DeviceId::cuda(0)));
    EXPECT_FALSE(result.hasP2P(DeviceId::cuda(1)));

    // Add P2P to GPU 1
    result.peer_transfer_gbps[DeviceId::cuda(1)] = 50.0;

    EXPECT_FALSE(result.hasP2P(DeviceId::cuda(0)));
    EXPECT_TRUE(result.hasP2P(DeviceId::cuda(1)));
}
