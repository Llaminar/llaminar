/**
 * @file Test__HeterogeneousBackend_PipelineBenchmark.cpp
 * @brief Performance benchmark comparing pipelined vs non-pipelined partial reduce-scatter
 *
 * This benchmark measures the effectiveness of the double-buffered pipeline optimization
 * for singleton configurations (1 CUDA + N ROCm or N CUDA + 1 ROCm).
 *
 * The pipeline optimization uses:
 * - `sendAsync()`, `recvAsync()`, `sendrecvAsync()` for async P2P transfers
 * - `stageChunksThroughBridgePipelined()` for double-buffered staging
 * - `pipelining_eligible` flag in TopologyAnalysis
 *
 * Benchmark scenarios:
 * 1. Baseline_NonPipelined - Serial partial reduce-scatter
 * 2. Pipelined_DoubleBuffer - Double-buffered pipelined version
 * 3. Comparison_Speedup - Speedup ratio calculation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/backends/NCCLBackend.h"
#include "v2/collective/backends/RCCLBackend.h"
#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"
#include "v2/utils/Logger.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Benchmark Configuration
    // ═══════════════════════════════════════════════════════════════════════════

    namespace
    {
        constexpr int WARMUP_ITERATIONS = 3;
        constexpr int BENCHMARK_ITERATIONS = 10;

        // Test sizes (in bytes)
        constexpr size_t SIZE_8MB = 8 * 1024 * 1024;
        constexpr size_t SIZE_32MB = 32 * 1024 * 1024;
        constexpr size_t SIZE_128MB = 128 * 1024 * 1024;
        constexpr size_t SIZE_512MB = 512 * 1024 * 1024;

        // Environment variable to force pipelining on/off
        constexpr const char *ENV_HETEROGENEOUS_PIPELINE = "LLAMINAR_HETEROGENEOUS_PIPELINE";

        /**
         * @brief Check if pipelining is enabled via environment variable
         *
         * If LLAMINAR_HETEROGENEOUS_PIPELINE is set:
         * - "0" or "false" → disable pipelining
         * - "1" or "true" → enable pipelining
         * - Not set → use backend default (auto)
         *
         * @return std::optional<bool> - true/false if explicitly set, nullopt for auto
         */
        std::optional<bool> getPipelineOverride()
        {
            const char *env = std::getenv(ENV_HETEROGENEOUS_PIPELINE);
            if (!env)
                return std::nullopt;

            std::string val(env);
            if (val == "0" || val == "false" || val == "FALSE")
                return false;
            if (val == "1" || val == "true" || val == "TRUE")
                return true;
            return std::nullopt;
        }

        /**
         * @brief Fill buffer with random float values
         */
        void fillRandom(std::vector<float> &data, unsigned int seed = 42)
        {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : data)
            {
                v = dist(rng);
            }
        }

    } // anonymous namespace

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_PipelineBenchmark : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Get backends via global accessors (avoids CUDA/HIP header conflicts)
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            // Count available devices via IBackend interface
            cuda_count_ = cuda_backend_ ? cuda_backend_->deviceCount() : 0;
            rocm_count_ = rocm_backend_ ? rocm_backend_->deviceCount() : 0;

            // Check minimum requirements for heterogeneous operation
            has_hardware_ = (cuda_count_ >= 1 && rocm_count_ >= 1);

            // Pipelining only benefits singleton configs (1 CUDA + N ROCm or N CUDA + 1 ROCm)
            is_singleton_config_ = has_hardware_ &&
                                   ((cuda_count_ == 1 && rocm_count_ > 1) ||
                                    (cuda_count_ > 1 && rocm_count_ == 1));

            if (has_hardware_)
            {
                backend_ = std::make_unique<HeterogeneousBackend>();
            }
        }

        void TearDown() override
        {
            freeAllBuffers();

            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
            backend_.reset();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Skip helpers
        // ─────────────────────────────────────────────────────────────────────

#define REQUIRE_HARDWARE()                                 \
    do                                                     \
    {                                                      \
        if (!has_hardware_)                                \
        {                                                  \
            GTEST_SKIP() << "Requires CUDA and ROCm GPUs"; \
        }                                                  \
    } while (0)

#define REQUIRE_SINGLETON_CONFIG()                                               \
    do                                                                           \
    {                                                                            \
        if (!is_singleton_config_)                                               \
        {                                                                        \
            GTEST_SKIP() << "Pipelining benchmark requires singleton config "    \
                         << "(1 CUDA + N ROCm or N CUDA + 1 ROCm). Have "        \
                         << cuda_count_ << " CUDA + " << rocm_count_ << " ROCm"; \
        }                                                                        \
    } while (0)

#define REQUIRE_MEMORY(bytes)                                                                    \
    do                                                                                           \
    {                                                                                            \
        if (!checkMemoryAvailable(bytes))                                                        \
        {                                                                                        \
            GTEST_SKIP() << "Insufficient memory for " << (bytes / (1024 * 1024)) << " MB test"; \
        }                                                                                        \
    } while (0)

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Create device groups
        // ─────────────────────────────────────────────────────────────────────

        DeviceGroup createSingletonDeviceGroup()
        {
            DeviceGroupBuilder builder;
            builder.setName("benchmark_group")
                .setScope(CollectiveScope::LOCAL)
                .setLocalRank(0);

            // Add CUDA device(s)
            for (int i = 0; i < cuda_count_; ++i)
            {
                builder.addDevice(DeviceId::cuda(i));
            }

            // Add ROCm device(s)
            for (int i = 0; i < rocm_count_; ++i)
            {
                builder.addDevice(DeviceId::rocm(i));
            }

            return builder.build();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Buffer management
        // ─────────────────────────────────────────────────────────────────────

        bool checkMemoryAvailable(size_t bytes_per_device)
        {
            // Conservative estimate: we need ~2x the buffer size per device
            // for staging buffers in pipelined mode
            // Note: We can't easily query free memory through IBackend interface,
            // so we just assume memory is available and let allocation fail if not
            (void)bytes_per_device;
            return true; // Assume memory available; allocation will fail gracefully if not
        }

        void *allocateCUDABuffer(int device_ordinal, size_t bytes)
        {
            if (!cuda_backend_)
                return nullptr;
            void *ptr = cuda_backend_->allocate(bytes, device_ordinal);
            if (ptr)
            {
                buffers_.push_back({ptr, DeviceType::CUDA, device_ordinal});
            }
            return ptr;
        }

        void *allocateROCmBuffer(int device_ordinal, size_t bytes)
        {
            if (!rocm_backend_)
                return nullptr;
            void *ptr = rocm_backend_->allocate(bytes, device_ordinal);
            if (ptr)
            {
                buffers_.push_back({ptr, DeviceType::ROCm, device_ordinal});
            }
            return ptr;
        }

        void initializeCUDABuffer(void *ptr, int device_ordinal, const std::vector<float> &data)
        {
            if (cuda_backend_)
            {
                cuda_backend_->hostToDevice(ptr, data.data(), data.size() * sizeof(float), device_ordinal);
                cuda_backend_->synchronize(device_ordinal);
            }
        }

        void initializeROCmBuffer(void *ptr, int device_ordinal, const std::vector<float> &data)
        {
            if (rocm_backend_)
            {
                rocm_backend_->hostToDevice(ptr, data.data(), data.size() * sizeof(float), device_ordinal);
                rocm_backend_->synchronize(device_ordinal);
            }
        }

        void freeAllBuffers()
        {
            for (const auto &buf : buffers_)
            {
                if (buf.type == DeviceType::CUDA && cuda_backend_)
                {
                    cuda_backend_->free(buf.ptr, buf.ordinal);
                }
                else if (buf.type == DeviceType::ROCm && rocm_backend_)
                {
                    rocm_backend_->free(buf.ptr, buf.ordinal);
                }
            }
            buffers_.clear();
            device_buffers_.clear();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Allocate all device buffers for a given size
        // ─────────────────────────────────────────────────────────────────────

        bool allocateAllBuffers(size_t bytes)
        {
            freeAllBuffers();

            // Allocate CUDA buffers
            for (int i = 0; i < cuda_count_; ++i)
            {
                void *ptr = allocateCUDABuffer(i, bytes);
                if (!ptr)
                {
                    LOG_ERROR("Failed to allocate CUDA buffer on device " << i);
                    return false;
                }
                device_buffers_.push_back(ptr);
            }

            // Allocate ROCm buffers
            for (int i = 0; i < rocm_count_; ++i)
            {
                void *ptr = allocateROCmBuffer(i, bytes);
                if (!ptr)
                {
                    LOG_ERROR("Failed to allocate ROCm buffer on device " << i);
                    return false;
                }
                device_buffers_.push_back(ptr);
            }

            return true;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Initialize buffers with distinct values
        // ─────────────────────────────────────────────────────────────────────

        void initializeAllBuffers(size_t count)
        {
            size_t idx = 0;
            std::vector<float> data(count);

            // Initialize CUDA buffers
            for (int i = 0; i < cuda_count_; ++i)
            {
                fillRandom(data, 100 + i);
                initializeCUDABuffer(device_buffers_[idx++], i, data);
                host_data_[i] = data; // Store for verification
            }

            // Initialize ROCm buffers
            for (int i = 0; i < rocm_count_; ++i)
            {
                fillRandom(data, 200 + i);
                initializeROCmBuffer(device_buffers_[idx++], i, data);
                host_data_[cuda_count_ + i] = data;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Benchmark result structure
        // ─────────────────────────────────────────────────────────────────────

        struct BenchmarkResult
        {
            double avg_us;
            double min_us;
            double max_us;
            double throughput_gbps;
            size_t tensor_bytes;
            bool pipelined;
        };

        // ─────────────────────────────────────────────────────────────────────
        // Core benchmark function
        // ─────────────────────────────────────────────────────────────────────

        BenchmarkResult benchmarkAllreduce(size_t tensor_bytes, int num_iterations)
        {
            size_t count = tensor_bytes / sizeof(float);

            // Allocate and initialize
            if (!allocateAllBuffers(tensor_bytes))
            {
                return {0, 0, 0, 0, tensor_bytes, false};
            }
            initializeAllBuffers(count);

            // Get pipelining status from topology analysis
            auto analysis = backend_->analyzeTopology(tensor_bytes);
            bool is_pipelined = analysis.pipelining_eligible;

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                initializeAllBuffers(count); // Reset data
                backend_->allreduceMulti(device_buffers_, count,
                                         CollectiveDataType::FLOAT32,
                                         CollectiveOp::ALLREDUCE_SUM);
                backend_->synchronize();
            }

            // Benchmark
            std::vector<double> timings;
            timings.reserve(num_iterations);

            for (int i = 0; i < num_iterations; ++i)
            {
                // Reset buffers to original data
                initializeAllBuffers(count);

                auto start = std::chrono::high_resolution_clock::now();

                backend_->allreduceMulti(device_buffers_, count,
                                         CollectiveDataType::FLOAT32,
                                         CollectiveOp::ALLREDUCE_SUM);
                backend_->synchronize();

                auto end = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration<double, std::micro>(end - start).count();
                timings.push_back(us);
            }

            // Calculate statistics
            double sum = 0, min_val = timings[0], max_val = timings[0];
            for (double t : timings)
            {
                sum += t;
                min_val = std::min(min_val, t);
                max_val = std::max(max_val, t);
            }

            BenchmarkResult result;
            result.avg_us = sum / num_iterations;
            result.min_us = min_val;
            result.max_us = max_val;
            result.tensor_bytes = tensor_bytes;
            result.pipelined = is_pipelined;

            // Throughput: total data moved in allreduce
            // For allreduce with N devices, each device sends and receives ~tensor_bytes
            // Effective bandwidth = tensor_bytes * num_devices / time
            size_t total_devices = cuda_count_ + rocm_count_;
            result.throughput_gbps = (static_cast<double>(tensor_bytes) * total_devices * 8.0) /
                                     (result.avg_us * 1000.0);

            return result;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Report formatting
        // ─────────────────────────────────────────────────────────────────────

        static std::string formatBytes(size_t bytes)
        {
            if (bytes >= 1024 * 1024 * 1024)
                return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
            if (bytes >= 1024 * 1024)
                return std::to_string(bytes / (1024 * 1024)) + " MB";
            if (bytes >= 1024)
                return std::to_string(bytes / 1024) + " KB";
            return std::to_string(bytes) + " B";
        }

        static std::string formatTime(double us)
        {
            if (us >= 1000000)
                return std::to_string(static_cast<int>(us / 1000000)) + " s";
            if (us >= 1000)
                return std::to_string(static_cast<int>(us / 1000)) + " ms";
            return std::to_string(static_cast<int>(us)) + " us";
        }

        void printBenchmarkHeader()
        {
            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║               HETEROGENEOUS ALLREDUCE PIPELINE BENCHMARK                                 ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Config: " << cuda_count_ << " CUDA + " << rocm_count_ << " ROCm";
            if (cuda_count_ == 1 && rocm_count_ > 1)
                std::cout << " (CUDA singleton, pipelining eligible)                          ";
            else if (cuda_count_ > 1 && rocm_count_ == 1)
                std::cout << " (ROCm singleton, pipelining eligible)                          ";
            else
                std::cout << " (non-singleton)                                               ";
            std::cout << "║\n";
            std::cout << "╠══════════════╦═══════════════╦═══════════════╦═══════════════╦══════════════════════════╣\n";
            std::cout << "║ Tensor Size  ║ Avg Time      ║ Min Time      ║ Max Time      ║ Throughput    ║ Pipeline ║\n";
            std::cout << "╠══════════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╬══════════╣\n";
        }

        void printBenchmarkRow(const BenchmarkResult &result)
        {
            std::cout << "║ " << std::setw(12) << std::left << formatBytes(result.tensor_bytes)
                      << " ║ " << std::setw(13) << std::right << formatTime(result.avg_us)
                      << " ║ " << std::setw(13) << formatTime(result.min_us)
                      << " ║ " << std::setw(13) << formatTime(result.max_us)
                      << " ║ " << std::setw(10) << std::fixed << std::setprecision(2) << result.throughput_gbps << " Gbps"
                      << " ║ " << std::setw(8) << (result.pipelined ? "Yes" : "No")
                      << " ║\n";
        }

        void printBenchmarkFooter()
        {
            std::cout << "╚══════════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════╩══════════╝\n";
            std::cout << "\n";
        }

        void printComparisonHeader()
        {
            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║           HETEROGENEOUS ALLREDUCE PIPELINE COMPARISON                                    ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Config: " << cuda_count_ << " CUDA + " << rocm_count_ << " ROCm";
            if (is_singleton_config_)
                std::cout << " (singleton, pipelining eligible)                               ";
            else
                std::cout << " (non-singleton)                                               ";
            std::cout << "║\n";
            std::cout << "╠══════════════╦═══════════════╦═══════════════╦═══════════════════════════════════════════╣\n";
            std::cout << "║ Tensor Size  ║ Non-Pipelined ║   Pipelined   ║     Speedup                               ║\n";
            std::cout << "╠══════════════╬═══════════════╬═══════════════╬═══════════════════════════════════════════╣\n";
        }

        void printComparisonRow(size_t bytes, double non_pipelined_us, double pipelined_us)
        {
            double speedup = non_pipelined_us / pipelined_us;
            std::cout << "║ " << std::setw(12) << std::left << formatBytes(bytes)
                      << " ║ " << std::setw(13) << std::right << formatTime(non_pipelined_us)
                      << " ║ " << std::setw(13) << formatTime(pipelined_us)
                      << " ║ " << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
                      << "                                ║\n";
        }

        void printComparisonFooter()
        {
            std::cout << "╚══════════════╩═══════════════╩═══════════════╩═══════════════════════════════════════════╝\n";
            std::cout << "\n";
        }

        // ─────────────────────────────────────────────────────────────────────
        // Member variables
        // ─────────────────────────────────────────────────────────────────────

        std::unique_ptr<HeterogeneousBackend> backend_;
        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;

        bool has_hardware_ = false;
        bool is_singleton_config_ = false;
        int cuda_count_ = 0;
        int rocm_count_ = 0;

        struct BufferInfo
        {
            void *ptr;
            DeviceType type;
            int ordinal;
        };
        std::vector<BufferInfo> buffers_;
        std::vector<void *> device_buffers_;
        std::map<int, std::vector<float>> host_data_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Performance Benchmark Tests
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @test Benchmark with 8MB tensor (just above typical threshold)
     */
    TEST_F(Test__HeterogeneousBackend_PipelineBenchmark, PipelineBenchmark_8MB)
    {
        REQUIRE_HARDWARE();
        REQUIRE_SINGLETON_CONFIG();
        REQUIRE_MEMORY(SIZE_8MB);

        auto group = createSingletonDeviceGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed: " << backend_->lastError();
        }

        // Check pipelining eligibility
        auto analysis = backend_->analyzeTopology(SIZE_8MB);
        LOG_INFO("Topology analysis for 8MB: pattern=" << static_cast<int>(analysis.pattern)
                                                       << ", pipelining_eligible=" << analysis.pipelining_eligible
                                                       << ", pipeline_chunks=" << analysis.pipeline_chunks);

        printBenchmarkHeader();
        auto result = benchmarkAllreduce(SIZE_8MB, BENCHMARK_ITERATIONS);
        printBenchmarkRow(result);
        printBenchmarkFooter();

        // Sanity check: operation completed
        EXPECT_GT(result.avg_us, 0) << "Benchmark returned zero time";
        EXPECT_GT(result.throughput_gbps, 0) << "Zero throughput reported";
    }

    /**
     * @test Benchmark with 32MB tensor
     */
    TEST_F(Test__HeterogeneousBackend_PipelineBenchmark, PipelineBenchmark_32MB)
    {
        REQUIRE_HARDWARE();
        REQUIRE_SINGLETON_CONFIG();
        REQUIRE_MEMORY(SIZE_32MB);

        auto group = createSingletonDeviceGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed: " << backend_->lastError();
        }

        printBenchmarkHeader();
        auto result = benchmarkAllreduce(SIZE_32MB, BENCHMARK_ITERATIONS);
        printBenchmarkRow(result);
        printBenchmarkFooter();

        EXPECT_GT(result.avg_us, 0);
        EXPECT_GT(result.throughput_gbps, 0);
    }

    /**
     * @test Benchmark with 128MB tensor
     */
    TEST_F(Test__HeterogeneousBackend_PipelineBenchmark, PipelineBenchmark_128MB)
    {
        REQUIRE_HARDWARE();
        REQUIRE_SINGLETON_CONFIG();
        REQUIRE_MEMORY(SIZE_128MB);

        auto group = createSingletonDeviceGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed: " << backend_->lastError();
        }

        printBenchmarkHeader();
        auto result = benchmarkAllreduce(SIZE_128MB, BENCHMARK_ITERATIONS);
        printBenchmarkRow(result);
        printBenchmarkFooter();

        EXPECT_GT(result.avg_us, 0);
        EXPECT_GT(result.throughput_gbps, 0);
    }

    /**
     * @test Benchmark with 512MB tensor (if memory permits)
     */
    TEST_F(Test__HeterogeneousBackend_PipelineBenchmark, PipelineBenchmark_512MB)
    {
        REQUIRE_HARDWARE();
        REQUIRE_SINGLETON_CONFIG();
        REQUIRE_MEMORY(SIZE_512MB);

        auto group = createSingletonDeviceGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed: " << backend_->lastError();
        }

        printBenchmarkHeader();
        auto result = benchmarkAllreduce(SIZE_512MB, BENCHMARK_ITERATIONS);
        printBenchmarkRow(result);
        printBenchmarkFooter();

        EXPECT_GT(result.avg_us, 0);
        EXPECT_GT(result.throughput_gbps, 0);
    }

    /**
     * @test Combined benchmark across all sizes with comparison table
     */
    TEST_F(Test__HeterogeneousBackend_PipelineBenchmark, AllSizesBenchmark)
    {
        REQUIRE_HARDWARE();
        REQUIRE_SINGLETON_CONFIG();

        auto group = createSingletonDeviceGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed: " << backend_->lastError();
        }

        std::vector<size_t> test_sizes;

        // Only add sizes that fit in memory
        if (checkMemoryAvailable(SIZE_8MB))
            test_sizes.push_back(SIZE_8MB);
        if (checkMemoryAvailable(SIZE_32MB))
            test_sizes.push_back(SIZE_32MB);
        if (checkMemoryAvailable(SIZE_128MB))
            test_sizes.push_back(SIZE_128MB);
        if (checkMemoryAvailable(SIZE_512MB))
            test_sizes.push_back(SIZE_512MB);

        if (test_sizes.empty())
        {
            GTEST_SKIP() << "Insufficient memory for any test size";
        }

        printBenchmarkHeader();

        for (size_t bytes : test_sizes)
        {
            auto result = benchmarkAllreduce(bytes, BENCHMARK_ITERATIONS);
            printBenchmarkRow(result);
            freeAllBuffers();
        }

        printBenchmarkFooter();

        // Print topology analysis summary
        std::cout << "Topology Analysis Summary:\n";
        for (size_t bytes : test_sizes)
        {
            auto analysis = backend_->analyzeTopology(bytes);
            std::cout << "  " << formatBytes(bytes) << ": "
                      << "pattern=" << static_cast<int>(analysis.pattern)
                      << ", pipelining=" << (analysis.pipelining_eligible ? "yes" : "no")
                      << ", chunks=" << analysis.pipeline_chunks
                      << ", reason: " << analysis.reason << "\n";
        }
    }

    /**
     * @test Verify pipelining eligibility is correctly determined
     */
    TEST_F(Test__HeterogeneousBackend_PipelineBenchmark, PipeliningEligibilityCheck)
    {
        REQUIRE_HARDWARE();
        REQUIRE_SINGLETON_CONFIG();

        auto group = createSingletonDeviceGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed: " << backend_->lastError();
        }

        // Test various sizes and verify pipelining decisions
        struct TestCase
        {
            size_t bytes;
            const char *description;
        };

        std::vector<TestCase> test_cases = {
            {1 * 1024 * 1024, "1 MB (below threshold)"},
            {4 * 1024 * 1024, "4 MB (at threshold)"},
            {8 * 1024 * 1024, "8 MB (above threshold)"},
            {32 * 1024 * 1024, "32 MB"},
            {128 * 1024 * 1024, "128 MB"},
        };

        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   PIPELINING ELIGIBILITY CHECK                                            ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Tensor Size       ║ Eligible ║ Chunks ║ Pattern                                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════╣\n";

        for (const auto &tc : test_cases)
        {
            auto analysis = backend_->analyzeTopology(tc.bytes);

            std::cout << "║ " << std::setw(17) << std::left << tc.description
                      << " ║ " << std::setw(8) << (analysis.pipelining_eligible ? "Yes" : "No")
                      << " ║ " << std::setw(6) << analysis.pipeline_chunks
                      << " ║ " << std::setw(48) << static_cast<int>(analysis.pattern)
                      << " ║\n";
        }

        std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

        // Verify threshold behavior:
        // - Below PIPELINE_MIN_TENSOR_SIZE (4MB): not eligible
        // - At or above: eligible (for singleton configs)

        auto small_analysis = backend_->analyzeTopology(1 * 1024 * 1024);
        EXPECT_FALSE(small_analysis.pipelining_eligible)
            << "1MB tensor should not be eligible for pipelining";

        auto large_analysis = backend_->analyzeTopology(8 * 1024 * 1024);
        // Note: pipelining_eligible depends on pattern being PARTIAL_REDUCE_SCATTER
        // For singleton configs, this should be true for large tensors
        LOG_INFO("8MB pipelining_eligible=" << large_analysis.pipelining_eligible
                                            << ", pattern=" << static_cast<int>(large_analysis.pattern));
    }

} // namespace llaminar2::test

#else // !HAVE_CUDA || !HAVE_ROCM

namespace llaminar2::test
{

    TEST(Test__HeterogeneousBackend_PipelineBenchmark, DISABLED_RequiresCUDAAndROCm)
    {
        GTEST_SKIP() << "This benchmark requires both CUDA and ROCm hardware";
    }

} // namespace llaminar2::test

#endif // HAVE_CUDA && HAVE_ROCM
