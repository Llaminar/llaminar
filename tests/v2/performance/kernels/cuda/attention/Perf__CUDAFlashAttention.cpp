/**
 * @file Perf__CUDAFlashAttention.cpp
 * @brief Performance benchmarks for CUDA Flash Attention kernels
 *
 * Measures throughput (TFLOPS, tokens/sec) for:
 *   - FA3 pipelined prefill (SM 8.0+, warp specialization, cp.async)
 *   - WMMA prefill (SM 7.0+, Tensor Core)
 *   - Scalar FP32 prefill (fallback)
 *   - Flash Decoding (all variants)
 *
 * **Tested Configurations**:
 * - Sequence lengths: 128, 512, 2048, 8192
 * - Head dimensions: 64, 128
 * - GQA ratios: 1:1 (MHA), 4:1 (GQA), 8:1 (GQA Qwen)
 *
 * **Performance Metrics**:
 * - TFLOPS: FLOPs / time (attention complexity: 4 * seq^2 * head_dim * n_heads)
 * - Tokens/sec: seq_len / time_per_attention
 * - Memory bandwidth: data moved / time
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"
#include "utils/MPIContext.h"
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include <cuda_runtime.h>

// Include test utils
#include "../../../../utils/CUDATestUtils.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>

using namespace llaminar2;
using namespace llaminar2::cuda;
using namespace llaminar2::test::cuda;

namespace
{

    // ============================================================================
    // Performance Test Configuration
    // ============================================================================

    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int BENCHMARK_ITERATIONS = 100;

    /**
     * @brief Test configuration
     */
    struct BenchConfig
    {
        int seq_len;
        int kv_len; // For decode: historical context length
        int n_heads;
        int n_kv_heads;
        int head_dim;
        const char *description;

        int gqaRatio() const { return n_heads / n_kv_heads; }

        // Attention FLOPs: Q @ K^T = seq * kv_len * head_dim, Softmax = seq * kv_len,
        // Att @ V = seq * kv_len * head_dim. Total per head: ~4 * seq * kv_len * head_dim
        double computeTFLOPs() const
        {
            double flops_per_head = 4.0 * seq_len * kv_len * head_dim;
            return (flops_per_head * n_heads) / 1e12;
        }

        // Memory: Q, K, V reads + O write (simplified)
        double computeMemoryGB() const
        {
            size_t q_bytes = seq_len * n_heads * head_dim * sizeof(float);
            size_t k_bytes = kv_len * n_kv_heads * head_dim * sizeof(float);
            size_t v_bytes = kv_len * n_kv_heads * head_dim * sizeof(float);
            size_t o_bytes = seq_len * n_heads * head_dim * sizeof(float);
            return (q_bytes + k_bytes + v_bytes + o_bytes) / (1024.0 * 1024.0 * 1024.0);
        }
    };

    // ============================================================================
    // CUDA Flash Attention Performance Test Fixture
    // ============================================================================

    class CUDAFlashAttentionPerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check CUDA availability
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0)
            {
                GTEST_SKIP() << "No CUDA devices available";
            }

            // Initialize device manager
            DeviceManager::instance().initialize(-1);
            if (!DeviceManager::instance().has_gpu())
            {
                GTEST_SKIP() << "No CUDA GPU available";
            }

            // Get device properties
            cudaGetDeviceProperties(&device_props_, 0);
            compute_capability_ = device_props_.major * 10 + device_props_.minor;

            std::cout << "Device: " << device_props_.name
                      << " (SM " << device_props_.major << "." << device_props_.minor << ")"
                      << std::endl;

            // Create CUDA events for timing
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);

            // Initialize MPI context (single rank for local test)
            mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_SELF);

            // Initialize kernel
            kernel_ = std::make_unique<CUDAFlashAttentionKernelT<ActivationPrecision::FP32>>(0);
        }

        void TearDown() override
        {
            if (start_event_)
                cudaEventDestroy(start_event_);
            if (stop_event_)
                cudaEventDestroy(stop_event_);
            kernel_.reset();
            mpi_ctx_.reset();
        }

        /**
         * @brief Generate random FP32 data
         */
        std::vector<float> randomFP32(size_t count, float scale = 0.1f)
        {
            std::vector<float> data(count);
            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, scale);
            for (auto &v : data)
                v = dist(rng);
            return data;
        }

        /**
         * @brief Run prefill benchmark for a single configuration
         *
         * NOTE: Currently only head_dim=64 is supported by the CUDA Flash Attention kernels.
         * head_dim=128 support requires kernel modifications to handle larger shared memory
         * requirements and different WMMA fragment shapes.
         */
        void runPrefillBenchmark(const BenchConfig &cfg)
        {
            const int seq_len = cfg.seq_len;
            const int n_heads = cfg.n_heads;
            const int n_kv_heads = cfg.n_kv_heads;
            const int head_dim = cfg.head_dim;

            const size_t q_size = seq_len * n_heads * head_dim;
            const size_t kv_size = seq_len * n_kv_heads * head_dim;
            const size_t out_size = seq_len * n_heads * head_dim;

            // Generate host data
            auto Q_data = randomFP32(q_size);
            auto K_data = randomFP32(kv_size);
            auto V_data = randomFP32(kv_size);

            // Allocate device memory
            float *d_Q, *d_K, *d_V, *d_output;
            cudaMalloc(&d_Q, q_size * sizeof(float));
            cudaMalloc(&d_K, kv_size * sizeof(float));
            cudaMalloc(&d_V, kv_size * sizeof(float));
            cudaMalloc(&d_output, out_size * sizeof(float));

            // Copy data to device
            cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                kernel_->compute(
                    d_Q, d_K, d_V, d_output,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    true,                               // causal
                    -1,                                 // window_size
                    nullptr, nullptr, nullptr, nullptr, // workspace buffers
                    false,                              // use_bf16
                    mpi_ctx_.get(),
                    0 // device_idx
                );
            }
            cudaDeviceSynchronize();

            // Benchmark
            std::vector<float> times_ms;
            times_ms.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                cudaEventRecord(start_event_);
                kernel_->compute(
                    d_Q, d_K, d_V, d_output,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    true, -1, nullptr, nullptr, nullptr, nullptr, false,
                    mpi_ctx_.get(), 0);
                cudaEventRecord(stop_event_);
                cudaEventSynchronize(stop_event_);

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event_, stop_event_);
                times_ms.push_back(ms);
            }

            // Cleanup
            cudaFree(d_Q);
            cudaFree(d_K);
            cudaFree(d_V);
            cudaFree(d_output);

            // Calculate statistics
            std::sort(times_ms.begin(), times_ms.end());
            float median_ms = times_ms[times_ms.size() / 2];
            float min_ms = times_ms.front();
            float max_ms = times_ms.back();

            // Calculate metrics
            double tflops_achieved = cfg.computeTFLOPs() / (median_ms / 1000.0);
            double memory_gb_s = cfg.computeMemoryGB() / (median_ms / 1000.0);
            double tokens_per_sec = seq_len / (median_ms / 1000.0);

            // Print results
            printPrefillResult(cfg, median_ms, min_ms, max_ms, tflops_achieved, memory_gb_s, tokens_per_sec);
        }

        /**
         * @brief Run decode benchmark for a single configuration
         *
         * NOTE: Currently only head_dim=64 is supported by the CUDA Flash Attention kernels.
         */
        void runDecodeBenchmark(const BenchConfig &cfg)
        {
            const int seq_len = 1; // Decode: single token
            const int kv_len = cfg.kv_len;
            const int n_heads = cfg.n_heads;
            const int n_kv_heads = cfg.n_kv_heads;
            const int head_dim = cfg.head_dim;

            const size_t q_size = n_heads * head_dim;
            const size_t kv_size = kv_len * n_kv_heads * head_dim;
            const size_t out_size = n_heads * head_dim;

            // Generate host data
            auto Q_data = randomFP32(q_size);
            auto K_data = randomFP32(kv_size);
            auto V_data = randomFP32(kv_size);

            // Allocate device memory
            float *d_Q, *d_K, *d_V, *d_output;
            cudaMalloc(&d_Q, q_size * sizeof(float));
            cudaMalloc(&d_K, kv_size * sizeof(float));
            cudaMalloc(&d_V, kv_size * sizeof(float));
            cudaMalloc(&d_output, out_size * sizeof(float));

            // Copy data to device
            cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);

            // Warmup
            for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            {
                kernel_->compute_decode(
                    d_Q, d_K, d_V, d_output,
                    seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    true, // causal
                    0     // position_offset
                );
            }
            cudaDeviceSynchronize();

            // Benchmark
            std::vector<float> times_ms;
            times_ms.reserve(BENCHMARK_ITERATIONS);

            for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
            {
                cudaEventRecord(start_event_);
                kernel_->compute_decode(
                    d_Q, d_K, d_V, d_output,
                    seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                    true, 0 // causal, position_offset
                );
                cudaEventRecord(stop_event_);
                cudaEventSynchronize(stop_event_);

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start_event_, stop_event_);
                times_ms.push_back(ms);
            }

            // Cleanup
            cudaFree(d_Q);
            cudaFree(d_K);
            cudaFree(d_V);
            cudaFree(d_output);

            // Calculate statistics
            std::sort(times_ms.begin(), times_ms.end());
            float median_ms = times_ms[times_ms.size() / 2];
            float min_ms = times_ms.front();
            float max_ms = times_ms.back();

            // Calculate latency-focused metrics
            double latency_us = median_ms * 1000.0; // Convert to microseconds
            double tokens_per_sec = 1.0 / (median_ms / 1000.0);

            // Print results
            printDecodeResult(cfg, latency_us, min_ms * 1000.0, max_ms * 1000.0, tokens_per_sec);
        }

        void printTableHeader(const std::string &test_name)
        {
            std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
            std::cout << "║ " << std::setw(92) << std::left << test_name << "║" << std::endl;
            std::cout << "║ Device: " << std::setw(84) << std::left << device_props_.name << "║" << std::endl;
            std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════╣" << std::endl;
        }

        void printPrefillHeader()
        {
            std::cout << "│ Config                    │ Median(ms) │ Min(ms) │ Max(ms) │ TFLOPS │ GB/s   │ Tok/s    │" << std::endl;
            std::cout << "├───────────────────────────┼────────────┼─────────┼─────────┼────────┼────────┼──────────┤" << std::endl;
        }

        void printPrefillResult(const BenchConfig &cfg,
                                float median_ms, float min_ms, float max_ms,
                                double tflops, double gb_s, double tok_s)
        {
            std::stringstream config_str;
            config_str << "seq=" << std::setw(4) << cfg.seq_len
                       << " h=" << cfg.n_heads << "/" << cfg.n_kv_heads
                       << " d=" << cfg.head_dim;

            std::cout << "│ " << std::setw(25) << std::left << config_str.str()
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(3) << median_ms
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(3) << min_ms
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(3) << max_ms
                      << " │ " << std::setw(6) << std::fixed << std::setprecision(2) << tflops
                      << " │ " << std::setw(6) << std::fixed << std::setprecision(0) << gb_s
                      << " │ " << std::setw(8) << std::fixed << std::setprecision(0) << tok_s
                      << " │" << std::endl;
        }

        void printDecodeHeader()
        {
            std::cout << "│ Config (KV cache len)     │ Median(μs) │ Min(μs) │ Max(μs) │ Tok/s      │" << std::endl;
            std::cout << "├───────────────────────────┼────────────┼─────────┼─────────┼────────────┤" << std::endl;
        }

        void printDecodeResult(const BenchConfig &cfg,
                               double median_us, double min_us, double max_us, double tok_s)
        {
            std::stringstream config_str;
            config_str << "kv=" << std::setw(5) << cfg.kv_len
                       << " h=" << cfg.n_heads << "/" << cfg.n_kv_heads
                       << " d=" << cfg.head_dim;

            std::cout << "│ " << std::setw(25) << std::left << config_str.str()
                      << " │ " << std::setw(10) << std::right << std::fixed << std::setprecision(1) << median_us
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(1) << min_us
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(1) << max_us
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(0) << tok_s
                      << " │" << std::endl;
        }

        void printTableFooter()
        {
            std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
        }

    protected:
        cudaDeviceProp device_props_;
        int compute_capability_;
        cudaEvent_t start_event_ = nullptr;
        cudaEvent_t stop_event_ = nullptr;
        std::unique_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<CUDAFlashAttentionKernelT<ActivationPrecision::FP32>> kernel_;
    };

    // ============================================================================
    // Prefill Benchmarks
    // ============================================================================

    TEST_F(CUDAFlashAttentionPerf, Prefill_SmallSeqLen_Qwen05B)
    {
        printTableHeader("Flash Attention Prefill - Small Sequences (Qwen 0.5B config)");
        printPrefillHeader();

        // Qwen 0.5B: 14 heads, 2 KV heads, head_dim=64
        std::vector<BenchConfig> configs = {
            {128, 128, 14, 2, 64, "Short prompt"},
            {256, 256, 14, 2, 64, "Medium prompt"},
            {512, 512, 14, 2, 64, "Long prompt"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_MediumSeqLen_Qwen7B)
    {
        printTableHeader("Flash Attention Prefill - Medium Sequences (Qwen 7B config)");
        printPrefillHeader();

        // Qwen 7B: 32 heads, 8 KV heads (GQA 4:1), head_dim=128
        std::vector<BenchConfig> configs = {
            {512, 512, 32, 8, 128, "Medium prompt"},
            {1024, 1024, 32, 8, 128, "Long prompt"},
            {2048, 2048, 32, 8, 128, "Very long prompt"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_LongSeqLen_Qwen72B)
    {
        printTableHeader("Flash Attention Prefill - Long Sequences (Qwen 72B config)");
        printPrefillHeader();

        // Qwen 72B: 64 heads, 8 KV heads (GQA 8:1), head_dim=128
        std::vector<BenchConfig> configs = {
            {2048, 2048, 64, 8, 128, "Long context"},
            {4096, 4096, 64, 8, 128, "Very long context"},
            {8192, 8192, 64, 8, 128, "Maximum context"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Prefill_HeadDimComparison)
    {
        printTableHeader("Flash Attention Prefill - Head Dimension Comparison");
        printPrefillHeader();

        // Compare head_dim=64 vs head_dim=128
        std::vector<BenchConfig> configs = {
            {1024, 1024, 32, 8, 64, "head_dim=64"},
            {1024, 1024, 32, 8, 128, "head_dim=128"},
            {2048, 2048, 32, 8, 64, "head_dim=64 long"},
            {2048, 2048, 32, 8, 128, "head_dim=128 long"},
        };

        for (const auto &cfg : configs)
        {
            runPrefillBenchmark(cfg);
        }

        printTableFooter();
    }

    // ============================================================================
    // Decode Benchmarks (Latency-focused)
    // ============================================================================

    TEST_F(CUDAFlashAttentionPerf, Decode_VaryingKVCacheLen)
    {
        printTableHeader("Flash Decoding - Varying KV Cache Length (Qwen 0.5B)");
        printDecodeHeader();

        // Qwen 0.5B: 14 heads, 2 KV heads, head_dim=64
        std::vector<BenchConfig> configs = {
            {1, 64, 14, 2, 64, "Early decode"},
            {1, 256, 14, 2, 64, "Short context"},
            {1, 1024, 14, 2, 64, "Medium context"},
            {1, 4096, 14, 2, 64, "Long context"},
            {1, 8192, 14, 2, 64, "Maximum context"},
        };

        for (const auto &cfg : configs)
        {
            runDecodeBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Decode_GQAComparison)
    {
        printTableHeader("Flash Decoding - GQA Ratio Comparison (kv_len=2048)");
        printDecodeHeader();

        // Compare different GQA ratios
        std::vector<BenchConfig> configs = {
            {1, 2048, 32, 32, 128, "MHA 1:1"},
            {1, 2048, 32, 8, 128, "GQA 4:1"},
            {1, 2048, 64, 8, 128, "GQA 8:1"},
            {1, 2048, 32, 4, 128, "GQA 8:1 alt"},
        };

        for (const auto &cfg : configs)
        {
            runDecodeBenchmark(cfg);
        }

        printTableFooter();
    }

    TEST_F(CUDAFlashAttentionPerf, Decode_LargeModel_Qwen72B)
    {
        printTableHeader("Flash Decoding - Large Model (Qwen 72B)");
        printDecodeHeader();

        // Qwen 72B: 64 heads, 8 KV heads, head_dim=128
        std::vector<BenchConfig> configs = {
            {1, 512, 64, 8, 128, "Short generation"},
            {1, 2048, 64, 8, 128, "Medium generation"},
            {1, 8192, 64, 8, 128, "Long generation"},
            {1, 16384, 64, 8, 128, "Very long generation"},
        };

        for (const auto &cfg : configs)
        {
            runDecodeBenchmark(cfg);
        }

        printTableFooter();
    }

    // ============================================================================
    // Roofline Analysis
    // ============================================================================

    TEST_F(CUDAFlashAttentionPerf, RooflineAnalysis)
    {
        printTableHeader("Flash Attention Roofline Analysis");

        // Get SM count and theoretical peak
        int sm_count = device_props_.multiProcessorCount;
        // FP32 throughput per SM (typically 128 FP32 ops per clock per SM on Ampere)
        // Clock rate is in MHz, so we need to convert
        double base_clock_ghz = 1.5;              // Typical sustained clock for modern GPUs
        double fp32_ops_per_sm_per_cycle = 128.0; // Ampere has 128 FP32 cores per SM
        double theoretical_tflops = sm_count * fp32_ops_per_sm_per_cycle * base_clock_ghz / 1000.0;

        // Memory bandwidth (convert from MHz to GB/s)
        // memoryBusWidth is in bits, memoryClockRate is in kHz (need to check units)
        double memory_bw_gb_s = 936.0; // RTX 3090: ~936 GB/s

        std::cout << "│ Theoretical Performance (estimated for Ampere):                                             │" << std::endl;
        std::cout << "│   SM count: " << std::setw(80) << std::left
                  << std::to_string(sm_count) << "│" << std::endl;
        std::cout << "│   FP32 Compute: ~" << std::setw(75) << std::left
                  << (std::to_string(static_cast<int>(theoretical_tflops)) + " TFLOPS (rough estimate)") << "│" << std::endl;
        std::cout << "│   Memory BW: ~" << std::setw(77) << std::left
                  << (std::to_string(static_cast<int>(memory_bw_gb_s)) + " GB/s (RTX 3090 spec)") << "│" << std::endl;
        std::cout << "├──────────────────────────────────────────────────────────────────────────────────────────────┤" << std::endl;

        printPrefillHeader();

        // Representative config for roofline
        BenchConfig cfg = {2048, 2048, 32, 8, 128, "Roofline test"};
        runPrefillBenchmark(cfg);

        std::cout << "├──────────────────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ Arithmetic Intensity: " << std::fixed << std::setprecision(2)
                  << std::setw(69) << std::left
                  << (std::to_string((cfg.computeTFLOPs() * 1e12) / (cfg.computeMemoryGB() * 1e9)) + " FLOPS/byte")
                  << "│" << std::endl;

        printTableFooter();
    }

} // anonymous namespace

#else // !HAVE_CUDA

TEST(CUDAFlashAttentionPerf, NoCUDA)
{
    GTEST_SKIP() << "CUDA not available - skipping Flash Attention performance tests";
}

#endif // HAVE_CUDA
