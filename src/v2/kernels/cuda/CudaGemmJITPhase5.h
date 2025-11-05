/**
 * @file CudaGemmJITPhase5.h
 * @brief JIT compilation for Phase 5 GEMM kernels with CuTe
 *
 * Extends base JIT infrastructure to support Phase 5-specific parameters:
 * - Buffer stages (single/double/triple buffering)
 * - Streaming sub-tile size
 * - CuTe atom layouts
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include "CudaGemmConfigPhase5.h"
#include <cuda.h>
#include <nvrtc.h>
#include <map>
#include <string>
#include <vector>
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
        struct CompiledKernelPhase5
        {
            CUmodule module = nullptr;
            CUfunction function = nullptr;

            CompiledKernelPhase5() = default;
            CompiledKernelPhase5(CUmodule mod, CUfunction func)
                : module(mod), function(func) {}

            ~CompiledKernelPhase5()
            {
                if (module != nullptr)
                {
                    cuModuleUnload(module);
                }
            }

            // Move-only
            CompiledKernelPhase5(const CompiledKernelPhase5 &) = delete;
            CompiledKernelPhase5 &operator=(const CompiledKernelPhase5 &) = delete;

            CompiledKernelPhase5(CompiledKernelPhase5 &&other) noexcept
                : module(other.module), function(other.function)
            {
                other.module = nullptr;
                other.function = nullptr;
            }

            CompiledKernelPhase5 &operator=(CompiledKernelPhase5 &&other) noexcept
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

            bool isValid() const
            {
                return module != nullptr && function != nullptr;
            }
        };

        /**
         * @brief Phase 5 JIT compiler
         */
        class CudaGemmJITPhase5
        {
        public:
            /**
             * Get singleton instance
             */
            static CudaGemmJITPhase5 &instance();

            /**
             * Get compiled kernel for given Phase 5 configuration
             *
             * @param config Phase 5 kernel configuration
             * @return CUDA function pointer ready to launch
             * @throws std::runtime_error if compilation fails
             */
            CUfunction getKernel(const CudaGemmConfigPhase5 &config);

            /**
             * Precompile kernels for given configurations (async)
             */
            void precompile(const std::vector<CudaGemmConfigPhase5> &configs);

            /**
             * Clear memory cache
             */
            void clearMemoryCache();

            /**
             * Cache statistics
             */
            struct CacheStats
            {
                size_t memory_hits = 0;
                size_t compiles = 0;
                size_t compilation_failures = 0;
                size_t cache_size_bytes = 0;
            };

            CacheStats getStats() const;

        private:
            CudaGemmJITPhase5();
            ~CudaGemmJITPhase5() = default;

            // Compilation pipeline
            std::string generateKernelSource(const CudaGemmConfigPhase5 &config);
            CompiledKernelPhase5 compileKernel(const CudaGemmConfigPhase5 &config);

            // Caching
            std::string getCacheKey(const CudaGemmConfigPhase5 &config) const;

            // Utilities
            std::string getGPUArchitecture() const;

            // State
            std::map<std::string, std::shared_ptr<CompiledKernelPhase5>> kernel_cache_;
            std::string gpu_arch_;
            mutable std::mutex cache_mutex_;
            mutable CacheStats stats_;
        };

    } // namespace cuda
} // namespace llaminar2
