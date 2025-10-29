/**
 * @file GemmAutoTuner.h
 * @brief Auto-tuning framework for IQ4_NL quantized GEMM kernels
 *
 * This auto-tuner benchmarks multiple kernel configurations (different unroll factors,
 * prefetch distances, and tile sizes) for each unique tensor shape encountered at runtime.
 * It caches the optimal configuration and routes future calls to the best-performing kernel.
 *
 * @author David Sanftenberg
 * @date January 2025
 */

#pragma once

#include "../../tensors/TensorKernels.h" // For IBlockDecoder
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <mutex>
#include <cstdint>
#include <sstream>

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {

            // Make llaminar2::IBlockDecoder available in this namespace
            using llaminar2::IBlockDecoder;

            /**
             * @brief Configuration parameters for a GEMM kernel variant
             */
            struct GemmKernelConfig
            {
                int unroll_factor;   // Loop unroll: 4, 8, 16, 24, 32
                int prefetch_blocks; // Prefetch distance: 0, 3, 5, 7, 9
                int tile_m;          // M dimension tile size: 8, 16, 32, 64
                int tile_n;          // N dimension tile size: 4, 8, 16, 32

                // Identifier for this configuration
                std::string id() const
                {
                    std::ostringstream oss;
                    oss << "unroll" << unroll_factor
                        << "_prefetch" << prefetch_blocks
                        << "_tile" << tile_m << "x" << tile_n;
                    return oss.str();
                }

                bool operator==(const GemmKernelConfig &other) const
                {
                    return unroll_factor == other.unroll_factor &&
                           prefetch_blocks == other.prefetch_blocks &&
                           tile_m == other.tile_m &&
                           tile_n == other.tile_n;
                }
            };

            /**
             * @brief Benchmark result for a kernel configuration
             */
            struct BenchmarkResult
            {
                GemmKernelConfig config;
                double gflops;
                double time_ms;
                int iterations;

                bool operator<(const BenchmarkResult &other) const
                {
                    return gflops > other.gflops; // Higher GFLOPS is better
                }
            };

            /**
             * @brief Hash function for tensor shapes (for caching)
             */
            struct ShapeHash
            {
                std::size_t operator()(const std::tuple<int, int, int> &shape) const
                {
                    auto h1 = std::hash<int>{}(std::get<0>(shape));
                    auto h2 = std::hash<int>{}(std::get<1>(shape));
                    auto h3 = std::hash<int>{}(std::get<2>(shape));
                    return h1 ^ (h2 << 1) ^ (h3 << 2);
                }
            };

            /**
             * @brief Abstract interface for kernel variants
             */
            class IQuantizedGemmVariant
            {
            public:
                virtual ~IQuantizedGemmVariant() = default;

                /**
                 * @brief Get human-readable name for this variant
                 */
                virtual const char *name() const = 0;

                /**
                 * @brief Execute GEMM: C = alpha * A * B + beta * C
                 *
                 * @param A Input matrix [m, k] (FP32)
                 * @param B Weight matrix [k, n] (IQ4_NL quantized, accessed via decoder)
                 * @param C Output matrix [m, n] (FP32)
                 * @param m Number of rows in A and C
                 * @param n Number of columns in B and C
                 * @param k Number of columns in A and rows in B
                 * @param decoder Block decoder for quantized B matrix
                 * @param alpha Scaling factor for A*B
                 * @param beta Scaling factor for existing C values
                 * @return true on success, false on error
                 */
                virtual bool multiply(
                    const float *A,
                    float *C,
                    int m, int n, int k,
                    const IBlockDecoder *decoder,
                    float alpha = 1.0f,
                    float beta = 0.0f) = 0;

                /**
                 * @brief Get configuration for this variant
                 */
                virtual GemmKernelConfig config() const = 0;
            };

            /**
             * @brief Auto-tuner that benchmarks and selects optimal kernel variants
             */
            class GemmAutoTuner
            {
            public:
                /**
                 * @brief Get singleton instance
                 */
                static GemmAutoTuner &instance();

                /**
                 * @brief Register a kernel variant
                 *
                 * @param variant Unique pointer to kernel implementation
                 */
                void registerVariant(std::unique_ptr<IQuantizedGemmVariant> variant);

                /**
                 * @brief Get optimal kernel for a given tensor shape
                 *
                 * On first call for a shape, runs auto-tuning benchmarks.
                 * On subsequent calls, returns cached optimal variant.
                 *
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @return Pointer to optimal kernel variant (never null)
                 */
                IQuantizedGemmVariant *getOptimalKernel(int m, int n, int k);

                /**
                 * @brief Manually set optimal kernel for a shape (skip auto-tuning)
                 *
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @param config Configuration to use
                 */
                void setOptimalConfig(int m, int n, int k, const GemmKernelConfig &config);

                /**
                 * @brief Clear cached configurations (for testing)
                 */
                void clearCache();

                /**
                 * @brief Get benchmark results for a shape (if auto-tuned)
                 *
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @return Vector of benchmark results (empty if not yet tuned)
                 */
                std::vector<BenchmarkResult> getBenchmarkResults(int m, int n, int k) const;

                /**
                 * @brief Enable/disable auto-tuning (default: enabled)
                 *
                 * When disabled, uses first registered variant for all shapes.
                 */
                void setAutoTuningEnabled(bool enabled);

                /**
                 * @brief Set number of warmup iterations for benchmarking (default: 3)
                 */
                void setWarmupIterations(int iterations);

                /**
                 * @brief Set number of timed iterations for benchmarking (default: 10)
                 */
                void setTimedIterations(int iterations);

                /**
                 * @brief Set minimum benchmark time in milliseconds (default: 100ms)
                 *
                 * Auto-tuner will run enough iterations to exceed this time for stable results.
                 */
                void setMinBenchmarkTimeMs(double min_time_ms);

                // ========== Test-only API ==========

                /**
                 * @brief Get all available kernel variants (for testing)
                 *
                 * Returns a copy of all registered kernel configurations.
                 * Triggers lazy registration if not yet initialized.
                 *
                 * @return Vector of all available kernel configurations
                 */
                std::vector<GemmKernelConfig> getAvailableVariants();

                /**
                 * @brief Create a specific kernel variant by configuration (for testing)
                 *
                 * Allows tests to directly instantiate and validate specific kernel
                 * variants without going through auto-tuning.
                 *
                 * @param config Kernel configuration to create
                 * @param decoder IBlockDecoder for quantized weight access
                 * @return Unique pointer to the kernel variant, or nullptr if not found
                 */
                std::unique_ptr<IQuantizedGemmVariant> createVariant(
                    const GemmKernelConfig &config,
                    const IBlockDecoder *decoder);

            private:
                GemmAutoTuner(); // Singleton
                ~GemmAutoTuner() = default;

                // Prevent copying
                GemmAutoTuner(const GemmAutoTuner &) = delete;
                GemmAutoTuner &operator=(const GemmAutoTuner &) = delete;

                /**
                 * @brief Run auto-tuning benchmarks for a shape
                 *
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @return Best configuration found
                 */
                GemmKernelConfig autoTune(int m, int n, int k);

                /**
                 * @brief Benchmark a single kernel variant
                 *
                 * @param variant Kernel to benchmark
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @return Benchmark result
                 */
                BenchmarkResult benchmarkVariant(
                    IQuantizedGemmVariant *variant,
                    int m, int n, int k);

                /**
                 * @brief Allocate test data for benchmarking
                 */
                void allocateTestData(int m, int n, int k);

                /**
                 * @brief Free test data
                 */
                void freeTestData();

                // Registered kernel variants
                std::vector<std::unique_ptr<IQuantizedGemmVariant>> variants_;

                // Cache: (m, n, k) -> optimal config
                std::unordered_map<
                    std::tuple<int, int, int>,
                    GemmKernelConfig,
                    ShapeHash>
                    optimal_configs_;

                // Benchmark history: (m, n, k) -> vector of results
                std::unordered_map<
                    std::tuple<int, int, int>,
                    std::vector<BenchmarkResult>,
                    ShapeHash>
                    benchmark_history_;

                // Thread safety for cache updates
                mutable std::mutex cache_mutex_;

                // Auto-tuning parameters
                bool auto_tuning_enabled_ = true;
                int warmup_iterations_ = 3;
                int timed_iterations_ = 10;
                double min_benchmark_time_ms_ = 100.0;

                // Test data (allocated on-demand, reused across benchmarks)
                float *test_A_ = nullptr;
                float *test_C_ = nullptr;
                class MockBlockDecoder *test_decoder_ = nullptr;
                int allocated_m_ = 0;
                int allocated_n_ = 0;
                int allocated_k_ = 0;
            };

            /**
             * @brief Factory function to create auto-tuned quantized GEMM kernel
             *
             * Creates an ITensorGemm instance that automatically selects the optimal
             * variant for each tensor shape using the GemmAutoTuner. Variants are
             * registered lazily on first use and optimal selections are cached.
             *
             * Available variants (26 total):
             * - 12 AVX512 variants: 4×/8×/16× unroll × 4 tile sizes
             * - 12 AVX2 variants: Same configurations for CPUs without AVX512
             * - 2 legacy variants: cache_blocked, row_wise
             *
             * @param decoder IBlockDecoder for accessing quantized weight data
             * @return ITensorGemm instance that auto-selects optimal variant
             */
            std::unique_ptr<llaminar2::ITensorGemm> createAutoTunedGemm(const llaminar2::IBlockDecoder *decoder);

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
