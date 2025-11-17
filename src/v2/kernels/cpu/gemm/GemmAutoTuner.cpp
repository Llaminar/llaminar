/**
 * @file GemmAutoTuner.cpp
 * @brief Implementation of auto-tuning framework for IQ4_NL GEMM kernels
 *
 * @author David Sanftenberg
 * @date January 2025
 */

#include "GemmAutoTuner.h"
#include "GemmMicroKernelAdapter.h"         // For registerAllGemmVariants
#include "SmartGemmSearch.h"                // For intelligent variant filtering
#include "../../../tensors/TensorKernels.h" // For ITensorGemmTileDataProvider
#include "../../../utils/Logger.h"
#include "../../../utils/CPUFeatures.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cblas.h> // For cblas_sgemm (strided FP32 GEMM fallback)

namespace llaminar
{
    namespace v2
    {
        namespace kernels
        {

            /**
             * @brief Mock block gemmTileDataProvider for benchmarking (minimal overhead)
             */
            class MockTensorGemmTileDataProvider : public ITensorGemmTileDataProvider
            {
            public:
                MockTensorGemmTileDataProvider(size_t rows, size_t cols, size_t block_size)
                    : rows_(rows), cols_(cols), block_size_(block_size)
                {
                    // Allocate minimal mock data
                    size_t blocks_per_row = (cols + block_size - 1) / block_size;
                    data_.resize(rows * blocks_per_row * block_size, 0.0f);
                }

                void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
                {
                    // Simple memcpy from mock data (minimal overhead for benchmarking)
                    size_t blocks_per_row = (cols_ + block_size_ - 1) / block_size_;
                    size_t offset = (row_idx * blocks_per_row + k_block_offset) * block_size_;
                    std::memcpy(output, &data_[offset], block_size_ * sizeof(float));
                }

                const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
                {
                    size_t blocks_per_row = (cols_ + block_size_ - 1) / block_size_;
                    size_t offset = (row_idx * blocks_per_row + k_block_offset) * block_size_;
                    return &data_[offset];
                }

                size_t decoder_rows() const override { return rows_; }
                size_t decoder_cols() const override { return cols_; }
                size_t block_size() const override { return block_size_; }

            private:
                size_t rows_;
                size_t cols_;
                size_t block_size_;
                std::vector<float> data_;
            };

            // Singleton instance
            GemmAutoTuner &GemmAutoTuner::instance()
            {
                static GemmAutoTuner instance;
                return instance;
            }

            GemmAutoTuner::GemmAutoTuner()
            {
                // Check environment variable to disable auto-tuning
                const char *disable_env = std::getenv("LLAMINAR_DISABLE_GEMM_AUTOTUNE");
                if (disable_env && std::atoi(disable_env) != 0)
                {
                    auto_tuning_enabled_ = false;
                    LOG_INFO("GEMM auto-tuning disabled via LLAMINAR_DISABLE_GEMM_AUTOTUNE");
                }

                // Check for custom benchmark iterations
                const char *warmup_env = std::getenv("LLAMINAR_AUTOTUNE_WARMUP");
                if (warmup_env)
                {
                    warmup_iterations_ = std::atoi(warmup_env);
                    LOG_DEBUG("Auto-tune warmup iterations: " << warmup_iterations_);
                }

                const char *timed_env = std::getenv("LLAMINAR_AUTOTUNE_ITERATIONS");
                if (timed_env)
                {
                    timed_iterations_ = std::atoi(timed_env);
                    LOG_DEBUG("Auto-tune timed iterations: " << timed_iterations_);
                }

                const char *min_time_env = std::getenv("LLAMINAR_AUTOTUNE_MIN_TIME_MS");
                if (min_time_env)
                {
                    min_benchmark_time_ms_ = std::atof(min_time_env);
                    LOG_DEBUG("Auto-tune min benchmark time: " << min_benchmark_time_ms_ << " ms");
                }
            }

            void GemmAutoTuner::registerVariant(std::unique_ptr<IQuantizedGemmVariant> variant)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);

                auto config = variant->config();
                LOG_DEBUG("Registered GEMM variant: " << config.id());

                variants_.push_back(std::move(variant));
            }

            IQuantizedGemmVariant *GemmAutoTuner::getOptimalKernel(int m, int n, int k)
            {
                // Fast path: check cache first (no lock for read-only check)
                auto shape = std::make_tuple(m, n, k);

                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto it = optimal_configs_.find(shape);
                    if (it != optimal_configs_.end())
                    {
                        // Found cached config - find matching variant
                        const auto &cached_config = it->second;
                        for (auto &variant : variants_)
                        {
                            if (variant->config() == cached_config)
                            {
                                return variant.get();
                            }
                        }
                    }
                }

                // No cached config - need to auto-tune or use default
                if (!auto_tuning_enabled_ || variants_.empty())
                {
                    // Auto-tuning disabled or no variants - return first variant
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    if (variants_.empty())
                    {
                        LOG_ERROR("No GEMM variants registered!");
                        return nullptr;
                    }
                    return variants_[0].get();
                }

                // Run auto-tuning
                LOG_INFO("Auto-tuning GEMM kernel for shape [" << m << ", " << n << ", " << k << "]...");
                auto optimal_config = autoTune(m, n, k);

                // Find variant matching optimal config
                std::lock_guard<std::mutex> lock(cache_mutex_);
                for (auto &variant : variants_)
                {
                    if (variant->config() == optimal_config)
                    {
                        LOG_INFO("Selected optimal kernel: " << optimal_config.id()
                                                             << " for shape [" << m << ", " << n << ", " << k << "]");
                        return variant.get();
                    }
                }

                // Fallback (should never happen)
                LOG_WARN("Optimal config not found in variants, using first variant");
                return variants_[0].get();
            }

            void GemmAutoTuner::setOptimalConfig(int m, int n, int k, const GemmKernelConfig &config)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto shape = std::make_tuple(m, n, k);
                optimal_configs_[shape] = config;
                LOG_DEBUG("Manually set optimal config for [" << m << ", " << n << ", " << k
                                                              << "]: " << config.id());
            }

            void GemmAutoTuner::clearCache()
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                optimal_configs_.clear();
                benchmark_history_.clear();
                LOG_DEBUG("Cleared auto-tuner cache");
            }

            std::vector<BenchmarkResult> GemmAutoTuner::getBenchmarkResults(int m, int n, int k) const
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto shape = std::make_tuple(m, n, k);
                auto it = benchmark_history_.find(shape);
                if (it != benchmark_history_.end())
                {
                    return it->second;
                }
                return {};
            }

            void GemmAutoTuner::setAutoTuningEnabled(bool enabled)
            {
                auto_tuning_enabled_ = enabled;
                LOG_DEBUG("Auto-tuning " << (enabled ? "enabled" : "disabled"));
            }

            void GemmAutoTuner::setWarmupIterations(int iterations)
            {
                warmup_iterations_ = iterations;
            }

            void GemmAutoTuner::setTimedIterations(int iterations)
            {
                timed_iterations_ = iterations;
            }

            void GemmAutoTuner::setMinBenchmarkTimeMs(double min_time_ms)
            {
                min_benchmark_time_ms_ = min_time_ms;
            }

            GemmKernelConfig GemmAutoTuner::autoTune(int m, int n, int k)
            {
                LOG_INFO("Auto-tuning GEMM for [" << m << ", " << n << ", " << k
                                                  << "] - using smart search (max " << variants_.size() << " variants)");

                // ============================================================
                // Phase 1: Hierarchical filtering (972 -> ~10-50 candidates)
                // ============================================================

                // Step 1: Filter by problem size (eliminate unsuitable tiles)
                auto size_filtered = SmartGemmSearch::filterByProblemSize(variants_, m, n, k);

                if (size_filtered.empty())
                {
                    LOG_WARN("No variants passed size filtering, using fallback");
                    return variants_[0]->config();
                }

                // Step 2: Filter by ISA (select best ISA for this CPU)
                auto isa_filtered = SmartGemmSearch::filterByISA(size_filtered);

                if (isa_filtered.empty())
                {
                    LOG_WARN("No variants passed ISA filtering, using fallback");
                    return variants_[0]->config();
                }

                // Step 3: Rank by analytical performance model
                auto ranked = SmartGemmSearch::rankByPerformanceModel(isa_filtered, m, n, k);

                // Step 4: Select top N candidates for benchmarking
                // Read from environment or use default
                const char *max_bench_env = std::getenv("LLAMINAR_AUTOTUNE_MAX_CANDIDATES");
                size_t max_to_benchmark = max_bench_env ? std::atoi(max_bench_env) : 10;

                auto candidates = SmartGemmSearch::selectTopCandidates(ranked, max_to_benchmark);

                LOG_INFO("Smart search: " << variants_.size() << " total -> "
                                          << size_filtered.size() << " size-filtered -> "
                                          << isa_filtered.size() << " ISA-filtered -> "
                                          << candidates.size() << " top candidates to benchmark");

                // ============================================================
                // Phase 2: Benchmark top candidates only
                // ============================================================

                // Allocate test data once
                allocateTestData(m, n, k);

                std::vector<BenchmarkResult> results;
                results.reserve(candidates.size());

                for (auto *variant : candidates)
                {
                    auto result = benchmarkVariant(variant, m, n, k);
                    results.push_back(result);

                    LOG_DEBUG("  " << result.config.id() << ": "
                                   << result.gflops << " GFLOPS ("
                                   << result.time_ms << " ms, "
                                   << result.iterations << " iters)");
                }

                // Sort by GFLOPS (descending)
                std::sort(results.begin(), results.end());

                // Cache results
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto shape = std::make_tuple(m, n, k);
                    benchmark_history_[shape] = results;
                    optimal_configs_[shape] = results[0].config;
                }

                LOG_INFO("Auto-tuning complete. Best: " << results[0].config.id()
                                                        << " (" << results[0].gflops << " GFLOPS)"
                                                        << " - benchmarked " << candidates.size()
                                                        << " of " << variants_.size() << " variants");

                // Free test data
                freeTestData();

                return results[0].config;
            }

            BenchmarkResult GemmAutoTuner::benchmarkVariant(
                IQuantizedGemmVariant *variant,
                int m, int n, int k)
            {
                BenchmarkResult result;
                result.config = variant->config();

                // Warmup iterations
                for (int i = 0; i < warmup_iterations_; ++i)
                {
                    variant->multiply(test_A_, test_C_, m, n, k, test_decoder_, false); // transpose_B=false for IQ4_NL
                }

                // Timed iterations
                auto start = std::chrono::high_resolution_clock::now();
                int iterations = 0;
                double elapsed_ms = 0.0;

                // Run until we exceed minimum benchmark time
                do
                {
                    for (int i = 0; i < timed_iterations_; ++i)
                    {
                        variant->multiply(test_A_, test_C_, m, n, k, test_decoder_, false); // transpose_B=false for IQ4_NL
                    }
                    iterations += timed_iterations_;

                    auto end = std::chrono::high_resolution_clock::now();
                    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                } while (elapsed_ms < min_benchmark_time_ms_);

                // Calculate metrics
                result.iterations = iterations;
                result.time_ms = elapsed_ms / iterations;

                // GFLOPS = (2 * M * N * K) / (time_seconds * 1e9)
                double flops = 2.0 * m * n * k;
                double time_seconds = result.time_ms / 1000.0;
                result.gflops = flops / (time_seconds * 1e9);

                return result;
            }

            void GemmAutoTuner::allocateTestData(int m, int n, int k)
            {
                // Free existing data if size changed
                if (test_A_ && (m != allocated_m_ || k != allocated_k_))
                {
                    freeTestData();
                }

                if (!test_A_)
                {
                    // Allocate aligned memory (size must be multiple of alignment)
                    constexpr size_t ALIGNMENT = 64;
                    size_t size_A = m * k * sizeof(float);
                    size_t size_C = m * n * sizeof(float);
                    // Round up to multiple of alignment
                    size_A = ((size_A + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
                    size_C = ((size_C + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

                    test_A_ = static_cast<float *>(aligned_alloc(ALIGNMENT, size_A));
                    test_C_ = static_cast<float *>(aligned_alloc(ALIGNMENT, size_C));

                    // Initialize with random data (avoid cache prefetching patterns)
                    for (int i = 0; i < m * k; ++i)
                    {
                        test_A_[i] = static_cast<float>(rand()) / RAND_MAX;
                    }
                    std::memset(test_C_, 0, m * n * sizeof(float));

                    // Create mock gemmTileDataProvider (32-element blocks for IQ4_NL)
                    test_decoder_ = new MockTensorGemmTileDataProvider(n, k, 32);

                    allocated_m_ = m;
                    allocated_n_ = n;
                    allocated_k_ = k;
                }
            }

            void GemmAutoTuner::freeTestData()
            {
                if (test_A_)
                {
                    free(test_A_);
                    test_A_ = nullptr;
                }
                if (test_C_)
                {
                    free(test_C_);
                    test_C_ = nullptr;
                }
                if (test_decoder_)
                {
                    delete test_decoder_;
                    test_decoder_ = nullptr;
                }
                allocated_m_ = allocated_n_ = allocated_k_ = 0;
            }

            /**
             * @brief Auto-tuned GEMM kernel wrapper class
             */
            class AutoTunedGemmKernel : public llaminar2::ITensorGemm
            {
            public:
                explicit AutoTunedGemmKernel(const llaminar2::ITensorGemmTileDataProvider *gemmTileDataProvider)
                    : gemmTileDataProvider_(gemmTileDataProvider)
                {
                    ensureVariantsRegistered();
                }

                bool multiply(
                    const float *A, float *C,
                    int m, int n, int k,
                    bool transpose_B,
                    float alpha, float beta,
                    const llaminar2::MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    (void)mpi_ctx;
                    (void)device_idx;

                    if (!gemmTileDataProvider_)
                    {
                        return false;
                    }

                    // Validate dimensions
                    int expected_cols = transpose_B ? k : n;
                    if (static_cast<int>(gemmTileDataProvider_->decoder_cols()) != expected_cols)
                    {
                        return false;
                    }

                    // Use auto-tuner to select optimal variant
                    auto &tuner = GemmAutoTuner::instance();
                    auto *optimal = tuner.getOptimalKernel(m, n, k);

                    if (!optimal)
                    {
                        return false;
                    }

                    // Delegate to auto-selected variant
                    return optimal->multiply(A, C, m, n, k, gemmTileDataProvider_, alpha, beta);
                }

                bool multiply_activations(
                    const float *A, const float *B, float *C,
                    int m, int n, int k,
                    bool transpose_B,
                    float alpha, float beta,
                    const llaminar2::MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    // TODO: Implement quantized activation-activation GEMM
                    // For now, unsupported (quantized tensors only support weight GEMM)
                    (void)A;
                    (void)B;
                    (void)C;
                    (void)m;
                    (void)n;
                    (void)k;
                    (void)transpose_B;
                    (void)alpha;
                    (void)beta;
                    (void)mpi_ctx;
                    (void)device_idx;
                    return false;
                }

                bool multiply_activations_strided(
                    const float *A, const float *B, float *C,
                    int m, int n, int k,
                    int lda, int ldb, int ldc,
                    bool transpose_B,
                    float alpha, float beta,
                    const llaminar2::MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    // Fallback: use OpenBLAS cblas_sgemm for strided FP32×FP32 GEMM
                    // This is used by CPUAttentionT for Q@K^T and weights@V computations
                    (void)mpi_ctx;    // Unused (single-node attention)
                    (void)device_idx; // CPU only

                    if (device_idx != -1)
                    {
                        return false; // Only CPU supported
                    }

                    // cblas_sgemm signature:
                    // void cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
                    // C = alpha * op(A) * op(B) + beta * C
                    // where op(X) = X or X^T based on Trans flags

                    const CBLAS_ORDER order = CblasRowMajor;
                    const CBLAS_TRANSPOSE trans_a = CblasNoTrans; // A is not transposed
                    const CBLAS_TRANSPOSE trans_b = transpose_B ? CblasTrans : CblasNoTrans;

                    cblas_sgemm(order, trans_a, trans_b,
                                m, n, k,
                                alpha,
                                A, lda,
                                B, ldb,
                                beta,
                                C, ldc);

                    return true;
                }

                bool supports_device(int device_idx) const override
                {
                    return device_idx == -1; // CPU only
                }

            private:
                const llaminar2::ITensorGemmTileDataProvider *gemmTileDataProvider_;

                void ensureVariantsRegistered()
                {
                    static bool registered = false;
                    static std::mutex registration_mutex;

                    if (!registered)
                    {
                        std::lock_guard<std::mutex> lock(registration_mutex);
                        if (!registered)
                        {
                            auto variants = llaminar2::kernels::gemm::registerMicroKernelVariants(gemmTileDataProvider_);
                            auto &tuner = GemmAutoTuner::instance();

                            for (auto &variant : variants)
                            {
                                tuner.registerVariant(std::move(variant));
                            }

                            registered = true;
                        }
                    }
                }
            };

            std::unique_ptr<llaminar2::ITensorGemm> createAutoTunedGemm(const llaminar2::ITensorGemmTileDataProvider *gemmTileDataProvider)
            {
                return std::make_unique<AutoTunedGemmKernel>(gemmTileDataProvider);
            }

            // ========== Test-only API Implementation ==========

            std::vector<GemmKernelConfig> GemmAutoTuner::getAvailableVariants()
            {
                // Trigger lazy registration by creating a dummy auto-tuned kernel
                // This ensures variants are registered before we query them
                static bool initialized = false;
                if (!initialized)
                {
                    // Create minimal mock gemmTileDataProvider to trigger registration
                    class TinyDecoder : public ITensorGemmTileDataProvider
                    {
                    public:
                        void decode_block_at(size_t, size_t, float *) const override {}
                        const void *get_raw_block_at(size_t, size_t) const override { return nullptr; }
                        size_t decoder_rows() const override { return 1; }
                        size_t decoder_cols() const override { return 32; }
                        size_t block_size() const override { return 32; }
                    };
                    TinyDecoder gemmTileDataProvider;
                    auto dummy = createAutoTunedGemm(&gemmTileDataProvider);
                    initialized = true;
                }

                std::lock_guard<std::mutex> lock(cache_mutex_);

                std::vector<GemmKernelConfig> configs;
                configs.reserve(variants_.size());

                for (const auto &variant : variants_)
                {
                    configs.push_back(variant->config());
                }

                return configs;
            }

            std::unique_ptr<IQuantizedGemmVariant> GemmAutoTuner::createVariant(
                const GemmKernelConfig &config,
                const ITensorGemmTileDataProvider *gemmTileDataProvider)
            {
                // Register all variants from scratch with the provided gemmTileDataProvider
                auto all_variants = llaminar2::kernels::gemm::registerMicroKernelVariants(gemmTileDataProvider);

                // Find matching variant
                for (auto &variant : all_variants)
                {
                    if (variant->config() == config)
                    {
                        return std::move(variant);
                    }
                }

                // Not found
                return nullptr;
            }

        } // namespace kernels
    } // namespace v2
} // namespace llaminar
