/**
 * @file CudaGemmJIT.h
 * @brief JIT compilation of CUDA GEMM kernels using NVRTC with disk caching
 *
 * This class manages runtime compilation of CUDA kernels with the following features:
 * - Just-in-time compilation via NVRTC (NVIDIA Runtime Compiler)
 * - Persistent disk cache (~/.cache/llaminar/cuda_kernels/)
 * - In-memory cache for compiled modules
 * - Automatic cache invalidation on kernel source changes
 *
 * Performance characteristics:
 * - First compilation: ~500ms per config
 * - Cache hit (disk): ~10ms to load module
 * - Cache hit (memory): <1ms (function pointer lookup)
 */

#pragma once

#include "CudaGemmConfig.h"
#include <cuda.h>
#include <nvrtc.h>
#include <map>
#include <string>
#include <vector> // Required for NVCC to parse method signatures
#include <optional>
#include <memory>
#include <mutex>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * Compiled kernel module with RAII cleanup
         */
        struct CompiledKernel
        {
            CUmodule module = nullptr;
            CUfunction function = nullptr;

            CompiledKernel() = default;

            CompiledKernel(CUmodule mod, CUfunction func)
                : module(mod), function(func) {}

            ~CompiledKernel()
            {
                if (module != nullptr)
                {
                    cuModuleUnload(module);
                }
            }

            // Move-only (no copying CUDA modules)
            CompiledKernel(const CompiledKernel &) = delete;
            CompiledKernel &operator=(const CompiledKernel &) = delete;

            CompiledKernel(CompiledKernel &&other) noexcept
                : module(other.module), function(other.function)
            {
                other.module = nullptr;
                other.function = nullptr;
            }

            CompiledKernel &operator=(CompiledKernel &&other) noexcept
            {
                if (this != &other)
                {
                    if (module != nullptr)
                    {
                        cuModuleUnload(module);
                    }
                    module = other.module;
                    function = other.function;
                    other.module = nullptr;
                    other.function = nullptr;
                }
                return *this;
            }

            /**
             * Check if kernel is valid (compiled successfully)
             */
            bool isValid() const
            {
                return module != nullptr && function != nullptr;
            }
        };

        /**
         * JIT compiler for CUDA GEMM kernels
         *
         * Thread-safe singleton that compiles kernels on-demand and caches results.
         */
        class CudaGemmJIT
        {
        public:
            /**
             * Get singleton instance
             */
            static CudaGemmJIT &instance();

            /**
             * Get compiled kernel for given configuration
             *
             * This method:
             * 1. Checks in-memory cache
             * 2. Checks disk cache
             * 3. Compiles via NVRTC if not cached
             * 4. Saves to disk cache for future runs
             *
             * @param config Kernel configuration (tile sizes, thread counts, etc.)
             * @return CUDA function pointer ready to launch
             * @throws std::runtime_error if compilation fails
             */
            CUfunction getKernel(const CudaGemmConfig &config);

            /**
             * Precompile kernels for given configurations (async)
             *
             * Useful for warming up the cache during initialization.
             *
             * @param configs List of configurations to precompile
             */
            void precompile(const std::vector<CudaGemmConfig> &configs);

            /**
             * Clear in-memory cache (disk cache persists)
             */
            void clearMemoryCache();

            /**
             * Clear disk cache (forces recompilation)
             */
            void clearDiskCache();

            /**
             * Get cache statistics
             */
            struct CacheStats
            {
                size_t memory_hits = 0;
                size_t disk_hits = 0;
                size_t compiles = 0;
                size_t compilation_failures = 0; // Count of kernels that failed due to resource constraints
                size_t cache_size_bytes = 0;
            };

            CacheStats getStats() const;

        private:
            CudaGemmJIT();
            ~CudaGemmJIT() = default;

            // No copying or moving
            CudaGemmJIT(const CudaGemmJIT &) = delete;
            CudaGemmJIT &operator=(const CudaGemmJIT &) = delete;

            /**
             * Compile kernel from source using NVRTC
             */
            CompiledKernel compileKernel(const CudaGemmConfig &config);

            /**
             * Generate specialized kernel source code
             */
            std::string generateKernelSource(const CudaGemmConfig &config);

            /**
             * Try to load compiled kernel from disk cache
             */
            std::optional<CompiledKernel> loadFromDiskCache(const CudaGemmConfig &config);

            /**
             * Save compiled kernel to disk cache
             */
            void saveToDiskCache(const CudaGemmConfig &config, const void *cubin, size_t size);

            /**
             * Get cache key for configuration
             */
            std::string getCacheKey(const CudaGemmConfig &config) const;

            /**
             * Get cache file path
             */
            std::string getCachePath(const CudaGemmConfig &config) const;

            /**
             * Get GPU architecture string (e.g., "sm_75")
             */
            std::string getGPUArchitecture() const;

            /**
             * Ensure cache directory exists
             */
            void ensureCacheDirectory();

            // In-memory cache: config -> compiled kernel
            mutable std::mutex cache_mutex_;
            std::map<std::string, std::shared_ptr<CompiledKernel>> kernel_cache_;

            // Cache directory path
            std::string cache_dir_;

            // GPU architecture (e.g., "sm_75")
            std::string gpu_arch_;

            // Statistics
            mutable CacheStats stats_;
        };

    } // namespace cuda
} // namespace llaminar2
