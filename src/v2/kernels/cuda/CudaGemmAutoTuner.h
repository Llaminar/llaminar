/**
 * @file CudaGemmAutoTuner.h
 * @brief Auto-tuning framework for CUDA GEMM kernels
 *
 * Similar to CPU GemmAutoTuner but adapted for CUDA-specific concerns:
 * - GPU occupancy and register pressure
 * - Shared memory bank conflicts
 * - Memory coalescing patterns
 * - Warp divergence
 *
 * Heuristic filtering reduces search space from ~200 variants to top 10-15
 * candidates before benchmarking.
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#pragma once

#include "CudaGemmConfig.h"
#include "IQ4_NL_BlockDecoder.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <tuple>
#include <mutex>
#include <string>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Benchmark result for a CUDA kernel configuration
         */
        struct CudaBenchmarkResult
        {
            CudaGemmConfig config;
            double gflops;
            double time_ms;
            int iterations;
            float occupancy;         // Measured occupancy
            int shared_memory_bytes; // Actual shared memory usage

            bool operator<(const CudaBenchmarkResult &other) const
            {
                return gflops > other.gflops; // Higher GFLOPS is better
            }
        };

        /**
         * @brief Hash function for tensor shapes
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
         * @brief Auto-tuner for CUDA GEMM kernels (singleton)
         *
         * Workflow:
         * 1. First call for shape (m, n, k):
         *    - Generate candidate configurations
         *    - Apply heuristic filters (problem size, occupancy, memory)
         *    - Benchmark top 10-15 candidates
         *    - Cache optimal configuration
         * 2. Subsequent calls: Return cached optimal config
         *
         * Environment variables:
         * - LLAMINAR_DISABLE_CUDA_AUTOTUNE: Use heuristic selection only
         * - LLAMINAR_CUDA_AUTOTUNE_CANDIDATES: Max candidates to benchmark (default: 10)
         * - LLAMINAR_CUDA_AUTOTUNE_ITERATIONS: Benchmark iterations (default: 10)
         */
        class CudaGemmAutoTuner
        {
        public:
            /**
             * @brief Get singleton instance
             */
            static CudaGemmAutoTuner &instance();

            /**
             * @brief Get optimal configuration for a tensor shape
             *
             * On first call, runs auto-tuning. On subsequent calls, returns cached config.
             *
             * @param m Number of rows
             * @param n Number of columns
             * @param k Inner dimension
             * @return Optimal configuration
             */
            CudaGemmConfig getOptimalConfig(int m, int n, int k);

            /**
             * @brief Manually set optimal configuration (skip auto-tuning)
             */
            void setOptimalConfig(int m, int n, int k, const CudaGemmConfig &config);

            /**
             * @brief Clear cached configurations (for testing)
             */
            void clearCache();

            /**
             * @brief Get benchmark results for a shape (if auto-tuned)
             */
            std::vector<CudaBenchmarkResult> getBenchmarkResults(int m, int n, int k) const;

            /**
             * @brief Enable/disable auto-tuning (default: enabled)
             */
            void setAutoTuningEnabled(bool enabled);

            /**
             * @brief Set max candidates to benchmark (default: 10)
             */
            void setMaxCandidates(int max_candidates);

            /**
             * @brief Set benchmark iterations (default: 10)
             */
            void setBenchmarkIterations(int iterations);

            /**
             * @brief Set warmup iterations (default: 3)
             */
            void setWarmupIterations(int iterations);

            /**
             * @brief Rank configurations by analytical performance model
             *
             * Supports two heuristics (controlled by LLAMINAR_USE_ML_HEURISTIC):
             * - Manual heuristic (default, DEPRECATED): Hand-tuned weights (correlation: -12,000)
             * - ML heuristic (LLAMINAR_USE_ML_HEURISTIC=1): Data-driven weights from Gradient Boosting (correlation: +6,000)
             *
             * ML weights are learned from 7,776 benchmark measurements and exported in cuda_heuristic_weights.h
             *
             * @param candidates List of configs to rank
             * @param m Number of rows (batch size - DOMINANT factor in ML model: 78%)
             * @param n Number of columns
             * @param k Inner dimension
             * @return Configs sorted by predicted performance (best first)
             */
            std::vector<CudaGemmConfig> rankByPerformanceModel(
                const std::vector<CudaGemmConfig> &candidates,
                int m, int n, int k);

            /**
             * @brief Compute manual heuristic score for a configuration
             *
             * Fallback scoring function used when:
             * - ML lookup table doesn't have the config (rare)
             * - ML heuristic is disabled
             *
             * @param config Configuration to score
             * @param m Number of rows
             * @param n Number of columns
             * @param k Inner dimension
             * @return Heuristic score (higher is better)
             */
            double manualHeuristicScore(
                const CudaGemmConfig &config,
                int m, int n, int k);

            /**
             * @brief Get all available configurations (for testing)
             */
            std::vector<CudaGemmConfig> getAvailableConfigs();

        private:
            CudaGemmAutoTuner(); // Singleton
            ~CudaGemmAutoTuner();

            // Prevent copying
            CudaGemmAutoTuner(const CudaGemmAutoTuner &) = delete;
            CudaGemmAutoTuner &operator=(const CudaGemmAutoTuner &) = delete;

            /**
             * @brief Run auto-tuning for a shape
             */
            CudaGemmConfig autoTune(int m, int n, int k);

            /**
             * @brief Generate all candidate configurations
             *
             * Returns ~100-200 valid configurations spanning the parameter space.
             */
            std::vector<CudaGemmConfig> generateCandidates();

            /**
             * @brief Filter candidates by problem size
             *
             * Eliminates configurations where:
             * - Tile dimensions exceed matrix dimensions (wasteful)
             * - Thread block size inappropriate for problem size
             */
            std::vector<CudaGemmConfig> filterByProblemSize(
                const std::vector<CudaGemmConfig> &candidates,
                int m, int n, int k);

            /**
             * @brief Filter by GPU resource constraints
             *
             * Eliminates configurations with:
             * - Poor occupancy (< 25%)
             * - Excessive register pressure (> 64 regs/thread)
             * - Shared memory overflow (> 48KB)
             */
            std::vector<CudaGemmConfig> filterByResources(
                const std::vector<CudaGemmConfig> &candidates);

        private:
            // Moving rankByPerformanceModel to public section above

            /**
             * @brief Select top N candidates for benchmarking
             */
            std::vector<CudaGemmConfig> selectTopCandidates(
                const std::vector<CudaGemmConfig> &ranked,
                int max_candidates);

            /**
             * @brief Heuristic selection (no benchmarking)
             *
             * Used when auto-tuning disabled or for very small matrices.
             */
            CudaGemmConfig selectHeuristic(int m, int n, int k);

            /**
             * @brief Benchmark a single configuration
             */
            CudaBenchmarkResult benchmarkConfig(
                const CudaGemmConfig &config,
                int m, int n, int k);

            /**
             * @brief Allocate GPU memory for benchmarking
             */
            void allocateTestData(int m, int n, int k);

            /**
             * @brief Free GPU memory
             */
            void freeTestData();

            // Cache: (m, n, k) -> optimal config
            std::unordered_map<
                std::tuple<int, int, int>,
                CudaGemmConfig,
                ShapeHash>
                optimal_configs_;

            // Benchmark history: (m, n, k) -> results
            std::unordered_map<
                std::tuple<int, int, int>,
                std::vector<CudaBenchmarkResult>,
                ShapeHash>
                benchmark_history_;

            // Thread safety
            mutable std::mutex cache_mutex_;

            // Auto-tuning parameters
            bool auto_tuning_enabled_ = true;
            int max_candidates_ = 10;
            int warmup_iterations_ = 3;
            int benchmark_iterations_ = 10;

            // Test data (GPU memory)
            float *test_A_device_ = nullptr;
            IQ4_NLBlock *test_B_device_ = nullptr;
            float *test_C_device_ = nullptr;
            int allocated_m_ = 0;
            int allocated_n_ = 0;
            int allocated_k_ = 0;

            // CUDA stream for benchmarking
            cudaStream_t benchmark_stream_ = nullptr;

            // CUDA events for precise timing
            cudaEvent_t start_event_ = nullptr;
            cudaEvent_t stop_event_ = nullptr;

            // Device properties (cached)
            cudaDeviceProp device_props_;
            int device_id_ = 0;
        };

    } // namespace cuda
} // namespace llaminar2
