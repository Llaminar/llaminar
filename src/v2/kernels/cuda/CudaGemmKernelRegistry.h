/**
 * @file CudaGemmKernelRegistry.h
 * @brief Runtime dispatch registry for CUDA GEMM kernel variants
 *
 * Provides O(1) lookup of pre-compiled CUDA kernel variants based on
 * runtime configuration parameters. Mirrors the CPU MicroKernelRegistry pattern.
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#pragma once

#include "CudaGemmConfig.h"
#include "IQ4_NL_BlockDecoder.h"
#include <cuda_runtime.h>
#include <map>
#include <tuple>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief CUDA kernel launcher function signature
         *
         * Launches a specific kernel variant with given grid/block configuration.
         */
        using CudaKernelLauncher = cudaError_t (*)(
            const float *A,
            const IQ4_NLBlock *B_blocks,
            float *C,
            int m, int n, int k,
            dim3 gridDim,
            dim3 blockDim,
            cudaStream_t stream);

        /**
         * @brief Configuration key for registry lookup
         *
         * (tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n,
         *  prefetch, transpose, vectorize, atom_type, atom_layout_m, atom_layout_n, atom_layout_k)
         */
        using CudaKernelKey = std::tuple<int, int, int, int, int, int, int, int, bool, int, int, int, int, int>;

        /**
         * @brief CUDA kernel registry singleton
         *
         * Provides O(1) runtime dispatch to pre-compiled kernel variants.
         */
        class CudaGemmKernelRegistry
        {
        public:
            /**
             * @brief Get singleton instance
             */
            static CudaGemmKernelRegistry &instance()
            {
                static CudaGemmKernelRegistry registry;
                return registry;
            }

            /**
             * @brief Register a kernel variant
             *
             * Called by static constructors in generated instantiation files.
             */
            void register_kernel(
                int tile_m, int tile_n, int tile_k,
                int threads_m, int threads_n,
                int work_m, int work_n,
                int prefetch, bool transpose, int vectorize,
                int atom_type, int atom_layout_m, int atom_layout_n, int atom_layout_k,
                CudaKernelLauncher launcher)
            {
                CudaKernelKey key{tile_m, tile_n, tile_k, threads_m, threads_n,
                                  work_m, work_n, prefetch, transpose, vectorize,
                                  atom_type, atom_layout_m, atom_layout_n, atom_layout_k};
                kernels_[key] = launcher;
            }

            /**
             * @brief Look up kernel launcher by configuration
             *
             * @return Launcher function pointer, or nullptr if not found
             */
            CudaKernelLauncher get_launcher(const CudaGemmConfig &config) const
            {
                CudaKernelKey key{
                    config.tile_m, config.tile_n, config.tile_k,
                    config.threads_m, config.threads_n,
                    config.work_per_thread_m, config.work_per_thread_n,
                    config.prefetch_stages, config.transpose_smem, config.vectorize_load,
                    config.atom_type, config.atom_layout_m, config.atom_layout_n, config.atom_layout_k};

                auto it = kernels_.find(key);
                if (it != kernels_.end())
                {
                    return it->second;
                }
                return nullptr;
            }

            /**
             * @brief Get number of registered kernels
             */
            size_t size() const { return kernels_.size(); }

            // Disable copy/move
            CudaGemmKernelRegistry(const CudaGemmKernelRegistry &) = delete;
            CudaGemmKernelRegistry &operator=(const CudaGemmKernelRegistry &) = delete;

        private:
            CudaGemmKernelRegistry() = default;

            std::map<CudaKernelKey, CudaKernelLauncher> kernels_;
        };

    } // namespace cuda
} // namespace llaminar2
