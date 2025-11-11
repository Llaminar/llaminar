/**
 * @file IntegerGemmAutoTuner.h
 * @brief Auto-tuning framework for Q8_0 integer GEMM kernels
 *
 * Extends GemmAutoTuner to support integer-domain GEMM with Q8_0 activations.
 * This is a parallel system to the FP32 GEMM auto-tuner, but operates entirely
 * in quantized space.
 *
 * Key differences from FP32 auto-tuner:
 * - Cache key includes activation format (Q8_0 vs FP32)
 * - Benchmark data uses Q8_0 tensors (not FP32)
 * - Variant selection considers INT8 VNNI availability
 * - Performance metrics account for quantization overhead
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "IntegerGemmAdapter.h"
#include "GemmAutoTuner.h"
#include "../../../tensors/Tensors.h"
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <mutex>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Configuration for integer GEMM variant
             *
             * Extends GemmKernelConfig with integer-specific parameters.
             */
            struct IntegerGemmConfig
            {
                int unroll_factor;
                int prefetch_blocks;
                int tile_m;
                int tile_n;

                // Integer-specific flags
                bool use_vnni;             // Use AVX512-VNNI instructions
                bool pack_weights;         // Pre-pack weights to Q8_0
                bool reuse_q8_activations; // Cache Q8_0 activations across GEMMs

                std::string id() const
                {
                    std::ostringstream oss;
                    oss << "int8_unroll" << unroll_factor
                        << "_prefetch" << prefetch_blocks
                        << "_tile" << tile_m << "x" << tile_n;
                    if (use_vnni)
                        oss << "_vnni";
                    if (pack_weights)
                        oss << "_pack";
                    return oss.str();
                }
            };

            /**
             * @brief Benchmark result for integer GEMM
             */
            struct IntegerBenchmarkResult
            {
                IntegerGemmConfig config;
                double gflops;
                double time_ms;
                double memory_bandwidth_gb_s;
                int iterations;

                bool operator<(const IntegerBenchmarkResult &other) const
                {
                    // Prefer higher GFLOPS, but also consider memory bandwidth
                    // (important for memory-bound INT8 operations)
                    return (gflops * 0.7 + memory_bandwidth_gb_s * 0.3) >
                           (other.gflops * 0.7 + other.memory_bandwidth_gb_s * 0.3);
                }
            };

            /**
             * @brief Hash function for integer GEMM cache key
             *
             * Includes activation format in key (Q8_0 vs FP32).
             */
            struct IntegerShapeHash
            {
                std::size_t operator()(const std::tuple<int, int, int, std::string> &key) const
                {
                    auto h1 = std::hash<int>{}(std::get<0>(key));         // m
                    auto h2 = std::hash<int>{}(std::get<1>(key));         // n
                    auto h3 = std::hash<int>{}(std::get<2>(key));         // k
                    auto h4 = std::hash<std::string>{}(std::get<3>(key)); // format
                    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
                }
            };

            /**
             * @brief Auto-tuner for Q8_0 integer GEMM
             *
             * This is a separate auto-tuner from the FP32 GEMM auto-tuner to allow
             * independent configuration and benchmarking of integer GEMM kernels.
             *
             * Usage:
             * ```cpp
             * auto &tuner = IntegerGemmAutoTuner::instance();
             * auto gemm = tuner.getOptimalKernel(M, N, K, weight_format);
             * gemm->multiplyQ8_0(A_q8, C_q8, M, N, K);
             * ```
             */
            class IntegerGemmAutoTuner
            {
            public:
                /**
                 * @brief Get singleton instance
                 */
                static IntegerGemmAutoTuner &instance()
                {
                    static IntegerGemmAutoTuner instance;
                    return instance;
                }

                /**
                 * @brief Register an integer GEMM variant
                 */
                template <typename ISA, int MR, int NR, int UNROLL_K = 4, int PREFETCH_DIST = 2>
                void registerVariant()
                {
                    IntegerGemmConfig config;
                    config.unroll_factor = UNROLL_K;
                    config.prefetch_blocks = PREFETCH_DIST;
                    config.tile_m = MR;
                    config.tile_n = NR;
                    config.use_vnni = std::is_same<ISA, simd::AVX512VNNITag>::value;
                    config.pack_weights = true;          // Always pack for Q8_0 path
                    config.reuse_q8_activations = false; // TODO: implement caching

                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    registered_configs_.push_back(config);
                }

                /**
                 * @brief Get optimal kernel for given shape and weight format
                 *
                 * @param m Number of rows
                 * @param n Number of columns
                 * @param k Inner dimension
                 * @param weight_format Weight tensor type string (e.g., "IQ4_NL", "Q6_K")
                 * @return Best integer GEMM configuration
                 */
                IntegerGemmConfig getOptimalConfig(int m, int n, int k, const std::string &weight_format)
                {
                    auto key = std::make_tuple(m, n, k, weight_format);

                    // Check cache first
                    {
                        std::lock_guard<std::mutex> lock(cache_mutex_);
                        auto it = optimal_configs_.find(key);
                        if (it != optimal_configs_.end())
                        {
                            return it->second;
                        }
                    }

                    // Run auto-tuning
                    if (auto_tuning_enabled_)
                    {
                        auto optimal = autoTune(m, n, k, weight_format);

                        // Cache result
                        std::lock_guard<std::mutex> lock(cache_mutex_);
                        optimal_configs_[key] = optimal;
                        return optimal;
                    }

                    // Return default configuration if auto-tuning disabled
                    return getDefaultConfig();
                }

                /**
                 * @brief Create integer GEMM kernel with specific configuration
                 */
                template <typename ISA = simd::AVX512VNNITag>
                std::unique_ptr<ITensorGemm> createKernel(
                    const IntegerGemmConfig &config,
                    const Q8_0Tensor *A,
                    const TensorBase *B)
                {
                    // Dispatch to appropriate variant based on tile sizes
                    // For now, use default MR=8, NR=8
                    // TODO: Generate template instantiations for all configurations
                    return createIntegerGemmWithConfig<ISA, 8, 8, 4, 2>(A, B);
                }

                /**
                 * @brief Enable/disable auto-tuning
                 */
                void setAutoTuningEnabled(bool enabled)
                {
                    auto_tuning_enabled_ = enabled;
                }

                /**
                 * @brief Clear cached configurations
                 */
                void clearCache()
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    optimal_configs_.clear();
                    benchmark_history_.clear();
                }

                /**
                 * @brief Get benchmark results for a shape
                 */
                std::vector<IntegerBenchmarkResult> getBenchmarkResults(
                    int m, int n, int k, const std::string &weight_format) const
                {
                    auto key = std::make_tuple(m, n, k, weight_format);
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto it = benchmark_history_.find(key);
                    if (it != benchmark_history_.end())
                    {
                        return it->second;
                    }
                    return {};
                }

            private:
                IntegerGemmAutoTuner() = default;

                IntegerGemmConfig autoTune(int m, int n, int k, const std::string &weight_format)
                {
                    // TODO: Implement full auto-tuning
                    // For now, return default configuration
                    return getDefaultConfig();
                }

                IntegerGemmConfig getDefaultConfig() const
                {
                    IntegerGemmConfig config;
                    config.unroll_factor = 4;
                    config.prefetch_blocks = 2;
                    config.tile_m = 8;
                    config.tile_n = 8;
                    config.use_vnni = true;
                    config.pack_weights = true;
                    config.reuse_q8_activations = false;
                    return config;
                }

                std::vector<IntegerGemmConfig> registered_configs_;

                std::unordered_map<
                    std::tuple<int, int, int, std::string>,
                    IntegerGemmConfig,
                    IntegerShapeHash>
                    optimal_configs_;

                std::unordered_map<
                    std::tuple<int, int, int, std::string>,
                    std::vector<IntegerBenchmarkResult>,
                    IntegerShapeHash>
                    benchmark_history_;

                mutable std::mutex cache_mutex_;
                bool auto_tuning_enabled_ = true;
            };

            /**
             * @brief Register all integer GEMM variants
             *
             * This function registers common INT8 VNNI configurations for auto-tuning.
             * Configurations are selected to balance register pressure and cache usage.
             */
            inline void registerAllIntegerGemmVariants()
            {
                auto &tuner = IntegerGemmAutoTuner::instance();

                using ISA = simd::AVX512VNNITag;

                // Small tiles (low register pressure)
                tuner.registerVariant<ISA, 4, 4, 4, 2>();
                tuner.registerVariant<ISA, 4, 8, 4, 2>();
                tuner.registerVariant<ISA, 8, 4, 4, 2>();

                // Medium tiles (balanced)
                tuner.registerVariant<ISA, 8, 8, 4, 2>();
                tuner.registerVariant<ISA, 8, 16, 4, 2>();
                tuner.registerVariant<ISA, 16, 8, 4, 2>();

                // Large tiles (high register pressure)
                tuner.registerVariant<ISA, 16, 16, 4, 2>();
                tuner.registerVariant<ISA, 32, 8, 4, 2>();
                tuner.registerVariant<ISA, 8, 32, 4, 2>();

                // High unroll variants (for large K)
                tuner.registerVariant<ISA, 8, 8, 8, 3>();
                tuner.registerVariant<ISA, 8, 8, 16, 5>();
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
